"""SanitizePhase — input validation and clamping before PerceivePhase.

Invariant: output values are finite and bounded per-series history.
No prediction, no inhibition, no fusion — only sanitization.

Pipeline-phase implementation: receives and returns PipelineContext.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Protocol, Tuple

import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..orchestration.phases import PipelineContext


class SeriesStatisticsProvider(Protocol):
    """Protocol for retrieving per-series mean and std."""

    def get_statistics(self, series_id: str) -> Optional[Tuple[float, float]]:
        """Return (mean, std) for the series, or None if unavailable."""
        ...


class LocalWindowStatisticsProvider:
    """Fallback provider: computes mean/std from the current values window."""

    def get_statistics(self, series_id: str) -> Optional[Tuple[float, float]]:
        return None


@dataclass(frozen=True)
class SanitizeConfig:
    """Thresholds for sanitization — all configurable via constructor."""

    sigma_multiplier: float = 6.0
    min_window_size: int = 3
    fallback_sigma: float = 1.0  # used when no history and window too small


class SanitizePhase:
    """Clamp values to [mean - 6σ, mean + 6σ] using per-series history.

    Hard-stops on NaN/Inf: sets is_fallback=True and
    fallback_reason="nan_or_inf_rejected".
    """

    def __init__(
        self,
        config: Optional[SanitizeConfig] = None,
        statistics_provider: Optional[SeriesStatisticsProvider] = None,
    ) -> None:
        self._cfg = config or SanitizeConfig()
        self._stats_provider = statistics_provider or LocalWindowStatisticsProvider()

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Sanitize ctx.values and update ctx with results.

        Args:
            ctx: Pipeline context with raw values.

        Returns:
            Modified context with sanitized_values and sanitization_flags.
            If NaN/Inf detected, ctx.is_fallback=True and
            ctx.fallback_reason="nan_or_inf_rejected".
        """
        values = ctx.values
        series_id = ctx.series_id
        flags: List[str] = []

        if not values:
            return ctx.with_field(
                sanitized_values=None,
                sanitization_flags=flags,
            )

        # Hard-stop on NaN/Inf
        if any(not math.isfinite(v) for v in values):
            flags.append("nan_or_inf_rejected")
            logger.warning(
                "sanitize_nan_or_inf_rejected",
                extra={"series_id": series_id, "n_values": len(values)},
            )
            return ctx.with_field(
                sanitized_values=[],
                sanitization_flags=flags,
                is_fallback=True,
                fallback_reason="nan_or_inf_rejected",
            )

        sanitized, flags = self._sanitize_values(values, series_id, flags)

        return ctx.with_field(
            values=sanitized,  # downstream phases see sanitized values
            sanitized_values=sanitized,
            sanitization_flags=flags,
        )

    def _sanitize_values(
        self, values: List[float], series_id: str, flags: List[str]
    ) -> Tuple[List[float], List[str]]:
        """Clamp values to [mean - 6σ, mean + 6σ]."""
        mean, std = self._resolve_statistics(values, series_id)
        if std is None or std == 0.0:
            return values, flags

        lower = mean - self._cfg.sigma_multiplier * std
        upper = mean + self._cfg.sigma_multiplier * std

        sanitized: List[float] = []
        clamped_count = 0
        for v in values:
            if v < lower or v > upper:
                clamped = max(lower, min(upper, v))
                sanitized.append(clamped)
                clamped_count += 1
            else:
                sanitized.append(v)

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

        return sanitized, flags

    def _resolve_statistics(
        self, values: List[float], series_id: str
    ) -> Tuple[float, Optional[float]]:
        """Return (mean, std) from provider or local window."""
        stats = self._stats_provider.get_statistics(series_id)
        if stats is not None:
            return stats

        n = len(values)
        if n < self._cfg.min_window_size:
            # Insufficient data — cannot compute robust statistics
            return 0.0, None

        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        std = math.sqrt(variance) if variance > 0 else 0.0
        return mean, std
