"""SanitizePhase — sanitizes raw inputs with imputation, spike detection,
clamping, and a data_quality_score (0-1) that flows to PredictionReadinessGate.

Invariants: never raises; NaN/Inf → impute (when history available) or
per-value reject; clamp to 6σ via Redis/local-window bounds; two-sided CUSUM
(k=0.5σ, h=4σ) flags ramps; spike >5σ from history → ``spike_suspected`` flag.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

from .bounds_provider import (
    BoundsProvider,
    LocalWindowBoundsProvider,
    SeriesValuesBoundsProvider,
)
from .cusum import detect_ramp
from .imputer import LinearInterpolator, MedianImputer
from ..series_values import SeriesValuesStore

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..orchestration.phases import PipelineContext


@dataclass(frozen=True)
class SanitizeConfig:
    """Tunables. ``max_values_buffered`` caps the Redis rolling buffer."""

    sigma_multiplier: float = 6.0
    min_window_size: int = 3
    redis_min_samples: int = 20
    max_values_buffered: int = 500
    cusum_k_sigma_factor: float = 0.5
    cusum_h_sigma_factor: float = 4.0
    spike_sigma_threshold: float = 5.0


class SanitizePhase:
    """Pipeline phase [0]: sanitize raw inputs before any perception."""

    name = "sanitize"

    def __init__(
        self,
        config: Optional[SanitizeConfig] = None,
        *,
        series_values_store: Optional[SeriesValuesStore] = None,
        primary_provider: Optional[BoundsProvider] = None,
        fallback_provider: Optional[BoundsProvider] = None,
        imputer: Optional[LinearInterpolator] = None,
    ) -> None:
        self._cfg = config or SanitizeConfig()
        self._store = series_values_store
        self._linear_imputer = imputer or LinearInterpolator(min_history=3)
        self._median_imputer = MedianImputer(min_history=3)
        self._primary = primary_provider or (
            SeriesValuesBoundsProvider(
                series_values_store, min_samples=self._cfg.redis_min_samples
            )
            if series_values_store is not None
            else None
        )
        self._fallback = fallback_provider or LocalWindowBoundsProvider(
            min_window_size=self._cfg.min_window_size,
        )

    def execute(self, ctx: "PipelineContext") -> "PipelineContext":
        try:
            return self._execute(ctx)
        except Exception as exc:
            logger.warning(
                "sanitize_exception_swallowed",
                extra={"series_id": getattr(ctx, "series_id", ""), "error": str(exc)},
            )
            flags = list(getattr(ctx, "sanitization_flags", []))
            flags.append("sanitize_exception_swallowed")
            return ctx.with_field(sanitization_flags=flags)

    def _execute(self, ctx: "PipelineContext") -> "PipelineContext":
        values = ctx.values
        series_id = ctx.series_id
        flags: List[str] = list(getattr(ctx, "sanitization_flags", []))
        n_total = len(values)

        store, primary = self._runtime_providers(ctx)

        if not values:
            return ctx.with_field(
                sanitized_values=None,
                sanitization_flags=flags,
                data_quality_score=1.0,
            )

        # --- 1. Impute / reject NaN/Inf values ---
        sanitized: List[float] = []
        imputed_count = 0
        rejected_count = 0

        for i, v in enumerate(values):
            if math.isfinite(v):
                sanitized.append(v)
                continue

            # Try in-window linear interpolation first
            imputed_val = self._interpolate_in_window(values, i)
            if imputed_val is not None:
                sanitized.append(imputed_val)
                imputed_count += 1
                flags.append("value_imputed")
                continue

            # Fall back to history-based median imputation
            history = store.get_recent(series_id) if store else None
            if history and len(history) >= 3:
                try:
                    imputed_val = self._median_imputer.impute(v, history)
                    sanitized.append(imputed_val)
                    imputed_count += 1
                    flags.append("value_imputed_from_history")
                except ValueError:
                    rejected_count += 1
                    flags.append("value_rejected")
            else:
                rejected_count += 1
                flags.append("value_rejected")

        # If ALL values were rejected → hard-stop fallback
        if not sanitized:
            flags.append("nan_or_inf_rejected")
            logger.warning(
                "sanitize_all_values_rejected",
                extra={"series_id": series_id, "n_values": n_total},
            )
            return ctx.with_field(
                sanitized_values=[],
                sanitization_flags=flags,
                is_fallback=True,
                fallback_reason="nan_or_inf_rejected",
                data_quality_score=0.0,
            )

        # --- 2. Spike detection (5σ from recent history) ---
        spike_count = 0
        history = store.get_recent(series_id) if store else None
        if history and len(history) >= 3:
            mean_h = sum(history) / len(history)
            var_h = sum((x - mean_h) ** 2 for x in history) / len(history)
            if var_h > 0:
                std_h = math.sqrt(var_h)
                spike_threshold = self._cfg.spike_sigma_threshold * std_h
                for v in sanitized:
                    if abs(v - mean_h) > spike_threshold:
                        spike_count += 1
                        flags.append("spike_suspected")

        # --- 3. Clamping bounds (primary Redis → local window fallback) ---
        clamped_count = 0
        bounds = self._bounds(sanitized, series_id, primary)
        if bounds is None:
            flags.append("bounds_unavailable_skipped")
        else:
            lower, upper = bounds
            clamped = []
            for v in sanitized:
                cv = max(lower, min(upper, v))
                clamped.append(cv)
                if cv != v:
                    clamped_count += 1
            if clamped_count > 0:
                flags.append(f"value_clamped:{clamped_count}")
                logger.info(
                    "sanitize_values_clamped",
                    extra={
                        "series_id": series_id,
                        "clamped_count": clamped_count,
                        "bounds": (round(lower, 4), round(upper, 4)),
                    },
                )
            sanitized = clamped

        # --- 4. CUSUM ramp detection ---
        if detect_ramp(
            sanitized,
            k_sigma_factor=self._cfg.cusum_k_sigma_factor,
            h_sigma_factor=self._cfg.cusum_h_sigma_factor,
        ):
            flags.append("cusum_ramp_detected")

        # --- 5. data_quality_score ---
        n_problematic = imputed_count + rejected_count + spike_count + clamped_count
        data_quality_score = max(0.0, 1.0 - (n_problematic / max(n_total, 1)))

        # --- 6. Persist for future invocations ---
        if store is not None:
            store.append_many(series_id, sanitized)

        return ctx.with_field(
            values=sanitized,
            sanitized_values=sanitized,
            sanitization_flags=flags,
            data_quality_score=data_quality_score,
        )

    # -- helpers ------------------------------------------------------

    def _runtime_providers(self, ctx):
        store, primary = self._store, self._primary
        if store is None:
            cand = getattr(getattr(ctx, "orchestrator", None), "_series_values_store", None)
            if isinstance(cand, SeriesValuesStore):
                store = cand
        if primary is None and store is not None:
            primary = SeriesValuesBoundsProvider(store, min_samples=self._cfg.redis_min_samples)
        return store, primary

    def _bounds(self, values, series_id, primary):
        sm = self._cfg.sigma_multiplier
        if primary is not None:
            b = primary.get_bounds(series_id, values, sm)
            if b is not None:
                return b
        return self._fallback.get_bounds(series_id, values, sm)

    @staticmethod
    def _interpolate_in_window(values: List[float], idx: int) -> Optional[float]:
        """Linearly interpolate value at *idx* using nearest valid neighbours."""
        left_val = None
        for j in range(idx - 1, -1, -1):
            if math.isfinite(values[j]):
                left_val = values[j]
                break
        right_val = None
        for j in range(idx + 1, len(values)):
            if math.isfinite(values[j]):
                right_val = values[j]
                break

        if left_val is not None and right_val is not None:
            return (left_val + right_val) / 2.0
        if left_val is not None:
            return left_val
        if right_val is not None:
            return right_val
        return None
