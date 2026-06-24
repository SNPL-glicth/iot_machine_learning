"""PredictionReadinessGate — caps max_action based on data_quality_score.

Reads ``ctx.data_quality_score`` (set by SanitizePhase / BoundaryCheckPhase)
and sets ``ctx.max_action`` accordingly:

  * score >= 0.5  → max_action = "PREDICT"    (full pipeline, no cap)
  * 0.3 <= score < 0.5 → max_action = "INVESTIGATE"
  * score < 0.3   → max_action = "LOG_ONLY"   (short-circuit fallback)

When the gate caps to LOG_ONLY the pipeline short-circuits with a
fallback result; INVESTIGATE still allows the downstream phases to
run but later phases (e.g. ActionGuardPhase) must honour the cap.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import PipelineContext

logger = logging.getLogger(__name__)


class PredictionReadinessGate:
    """Gate that limits action capability based on overall data quality."""

    name = "prediction_readiness_gate"

    SCORE_THRESHOLD_INVESTIGATE = 0.5
    SCORE_THRESHOLD_LOG_ONLY = 0.3

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        score = getattr(ctx, "data_quality_score", 1.0)

        if score >= self.SCORE_THRESHOLD_INVESTIGATE:
            return ctx.with_field(
                max_action="PREDICT",
            )

        if score >= self.SCORE_THRESHOLD_LOG_ONLY:
            logger.info(
                "prediction_readiness_capped_to_investigate",
                extra={
                    "series_id": ctx.series_id,
                    "data_quality_score": round(score, 3),
                },
            )
            return ctx.with_field(
                max_action="INVESTIGATE",
            )

        logger.warning(
            "prediction_readiness_capped_to_log_only",
            extra={
                "series_id": ctx.series_id,
                "data_quality_score": round(score, 3),
            },
        )
        return ctx.with_field(
            max_action="LOG_ONLY",
            is_fallback=True,
            fallback_reason="quality_too_low_log_only",
        )
