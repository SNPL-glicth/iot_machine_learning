"""Boundary Check Phase — validates input data against domain boundaries.

ALWAYS produces a useful result:
  * If SensorProfile exists → validates against physical operational_range.
  * If no SensorProfile → computes dynamic range (p1-p99) from current
    values and uses that as boundary.
  * Returns boundary_result with within_domain, data_quality_score,
    dynamic_range_used, and warnings for near-boundary values.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from . import PipelineContext

from iot_machine_learning.domain.entities.results.boundary_result import BoundaryResult

logger = logging.getLogger(__name__)

BOUNDARY_WARNING_MARGIN = 0.05  # 5% margin for near-boundary warnings


class BoundaryCheckPhase:
    """Phase 1: Domain boundary validation (always active)."""

    @property
    def name(self) -> str:
        return "boundary_check"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            return self._execute(ctx)
        except Exception as exc:
            logger.debug("boundary_check_skipped", extra={"error": str(exc)})
            return ctx

    def _execute(self, ctx: PipelineContext) -> PipelineContext:
        values = ctx.values
        series_id = ctx.series_id

        if not values:
            return ctx.with_field(
                boundary_result=BoundaryResult(
                    within_domain=True,
                    data_quality_score=getattr(ctx, "data_quality_score", 1.0),
                    warnings=["empty_values"],
                ),
            )

        profile = ctx.profile

        if profile is not None and hasattr(profile, "operational_range"):
            lower, upper = profile.operational_range
            dynamic_used = False
        else:
            lower, upper = _percentile_range(values, p_low=1.0, p_high=99.0)
            padding = (upper - lower) * 0.10 if upper > lower else 1.0
            lower -= padding
            upper += padding
            dynamic_used = True

        if math.isinf(lower) or math.isinf(upper) or math.isnan(lower) or math.isnan(upper):
            lower = min(values) if values else 0.0
            upper = max(values) if values else 0.0
            if lower == upper:
                upper = lower + 1.0

        warnings: List[str] = []
        n_outside = 0
        n_near_lower = 0
        n_near_upper = 0

        margin = (upper - lower) * BOUNDARY_WARNING_MARGIN
        warn_lower = lower + margin
        warn_upper = upper - margin

        for v in values:
            if v < lower - 1e-12 or v > upper + 1e-12:
                n_outside += 1
            elif v < warn_lower:
                n_near_lower += 1
            elif v > warn_upper:
                n_near_upper += 1

        if n_outside > 0:
            warnings.append(f"values_outside_boundary:{n_outside}")

        if n_near_lower > 0:
            warnings.append(f"values_near_lower_boundary:{n_near_lower}")

        if n_near_upper > 0:
            warnings.append(f"values_near_upper_boundary:{n_near_upper}")

        within_domain = n_outside == 0

        # data_quality_score: 1.0 baseline, -0.1 per % of problematic values
        n_problematic = n_outside + n_near_lower + n_near_upper
        dq_score = max(0.0, 1.0 - (n_problematic / max(len(values), 1)))

        if not within_domain:
            logger.warning(
                "domain_boundary_violation",
                extra={
                    "series_id": series_id,
                    "n_outside": n_outside,
                    "boundary": (round(lower, 4), round(upper, 4)),
                    "dynamic_range_used": dynamic_used,
                },
            )

        # Take the minimum of existing data_quality_score and this phase's score
        existing_score = getattr(ctx, "data_quality_score", 1.0)
        final_dq_score = min(existing_score, dq_score)

        ctx = ctx.with_field(
            boundary_result=BoundaryResult(
                within_domain=within_domain,
                rejection_reason=None if within_domain else "out_of_domain",
                data_quality_score=dq_score,
                warnings=warnings,
                dynamic_range_used=dynamic_used,
            ),
            data_quality_score=final_dq_score,
        )

        if not within_domain:
            ctx = ctx.with_field(
                is_fallback=True,
                fallback_reason="out_of_domain",
            )

        return ctx

    def create_early_result(self, ctx: PipelineContext):
        from iot_machine_learning.infrastructure.ml.interfaces import PredictionResult
        boundary = ctx.boundary_result
        return PredictionResult(
            predicted_value=None,
            confidence=0.0,
            trend="unknown",
            metadata={
                "is_out_of_domain": True,
                "rejection_reason": boundary.rejection_reason,
                "boundary_check": {
                    "within_domain": False,
                    "rejection_reason": boundary.rejection_reason,
                    "data_quality_score": boundary.data_quality_score,
                    "warnings": boundary.warnings,
                    "dynamic_range_used": boundary.dynamic_range_used,
                },
            },
        )


def _percentile_range(
    values: List[float],
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> Tuple[float, float]:
    """Compute percentile-based range from *values*."""
    sorted_v = sorted(values)
    n = len(sorted_v)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (sorted_v[0], sorted_v[0] + 1.0)

    def _percentile(sorted_data: List[float], p: float) -> float:
        k = (p / 100.0) * (n - 1)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_data[int(k)]
        return sorted_data[int(f)] * (c - k) + sorted_data[int(c)] * (k - f)

    low = _percentile(sorted_v, p_low)
    high = _percentile(sorted_v, p_high)
    if high <= low:
        # Degenerate — widen artificially
        mid = (min(sorted_v) + max(sorted_v)) / 2.0
        half = max(abs(max(sorted_v) - mid), abs(mid - min(sorted_v)), 1.0)
        low = mid - half
        high = mid + half
    return (low, high)
