"""SanitizePhase — IMP-1 phase [0] of the cognitive pipeline.

Invariants: never raises; NaN/Inf → hard-stop fallback; clamp to 6σ
using Redis-backed history (:class:`SeriesValuesStore`) with local
window as fallback; two-sided CUSUM (k=0.5σ, h=4σ) flags ramps.
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
from .imputer import MedianImputer  # COG-CRIT-1
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
        imputer: Optional[MedianImputer] = None,  # COG-CRIT-1
    ) -> None:
        self._cfg = config or SanitizeConfig()
        self._store = series_values_store
        self._imputer = imputer or MedianImputer(min_history=3)  # COG-CRIT-1
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

    # -- main entry point ---------------------------------------------

    def execute(self, ctx: "PipelineContext") -> "PipelineContext":
        """Sanitize ``ctx.values``. Never raises."""
        try:
            return self._execute(ctx)
        except Exception as exc:  # pragma: no cover — defensive
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

        store, primary = self._runtime_providers(ctx)

        if not values:
            return ctx.with_field(sanitized_values=None, sanitization_flags=flags)

        # 1. COG-CRIT-1: Impute NaN/Inf values instead of rejecting entire window
        imputed_count = 0
        sanitized = []
        for v in values:
            if math.isfinite(v):
                sanitized.append(v)
            else:
                # Try to impute using historical data from store
                history = store.get_values(series_id) if store else None
                if history and len(history) >= 3:
                    try:
                        imputed = self._imputer.impute(v, history)
                        sanitized.append(imputed)
                        imputed_count += 1
                        flags.append(f"value_imputed:{imputed_count}")
                    except ValueError:
                        # Cannot impute - reject this value only
                        flags.append(f"value_rejected:insufficient_history")
                else:
                    # No history available - reject this value only
                    flags.append(f"value_rejected:no_history")
        
        # If all values were rejected, fallback
        if not sanitized:
            flags.append("all_values_rejected")
            logger.warning(
                "sanitize_all_values_rejected",
                extra={"series_id": series_id, "n_values": len(values)},
            )
            return ctx.with_field(
                sanitized_values=[],
                sanitization_flags=flags,
                is_fallback=True,
                fallback_reason="all_values_rejected",
            )
        
        # COG-CRIT-1: Apply penalty for imputed values
        if imputed_count > 0:
            penalty = imputed_count * 0.1
            confidence_multiplier = max(0.5, 1.0 - penalty)
            ctx = ctx.with_field(confidence_multiplier=confidence_multiplier)
            logger.info(
                "sanitize_values_imputed",
                extra={
                    "series_id": series_id,
                    "imputed_count": imputed_count,
                    "confidence_multiplier": round(confidence_multiplier, 3),
                },
            )

        # 2. Resolve clamping bounds (primary Redis → local window fallback).
        bounds = self._bounds(sanitized, series_id, primary)
        if bounds is None:
            flags.append("bounds_unavailable_skipped")
        else:
            sanitized, flags = self._clamp(sanitized, bounds, flags, series_id)

        # 3. CUSUM ramp detection (does not block).
        if detect_ramp(
            sanitized,
            k_sigma_factor=self._cfg.cusum_k_sigma_factor,
            h_sigma_factor=self._cfg.cusum_h_sigma_factor,
        ):
            flags.append("cusum_ramp_detected")

        # 4. Persist for future invocations.
        if store is not None:
            store.append_many(series_id, sanitized)

        return ctx.with_field(
            values=sanitized,
            sanitized_values=sanitized,
            sanitization_flags=flags,
        )

    # -- helpers ------------------------------------------------------

    def _runtime_providers(self, ctx):
        """Pick up store/primary from ctx.orchestrator (singleton path)."""
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
    def _clamp(values, bounds, flags, series_id):
        lower, upper = bounds
        sanitized = [max(lower, min(upper, v)) for v in values]
        clamped = sum(1 for s, v in zip(sanitized, values) if s != v)
        if clamped > 0:
            flags.append(f"value_clamped:{clamped}")
            logger.info(
                "sanitize_values_clamped",
                extra={"series_id": series_id, "clamped_count": clamped,
                       "bounds": (round(lower, 4), round(upper, 4))},
            )
        return sanitized, flags
