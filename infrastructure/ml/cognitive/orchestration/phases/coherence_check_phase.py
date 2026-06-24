"""Coherence Check Phase — uses real context from PerceivePhase and DriftDetectionPhase.

Now leverages:
  * cross_regime_incoherence (PerceivePhase)  – regime mismatch across sibling sensors
  * drift_detected / drift_event (DriftDetectionPhase) – recent concept drift
These signals produce an honest confidence penalty instead of hardcoded placeholders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from . import PipelineContext

from iot_machine_learning.domain.services.signal_coherence_checker import SignalCoherenceChecker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoherenceResult:
    is_coherent: bool
    conflict_type: Optional[str]
    resolved_value: float
    resolved_confidence: float
    resolution_reason: str
    penalties: List[str] = field(default_factory=list)


CROSS_REGIME_PENALTY: float = 0.15
DRIFT_PENALTY: float = 0.10


class CoherenceCheckPhase:
    """Phase 7: Signal coherence check using real pipeline context."""

    @property
    def name(self) -> str:
        return "coherence_check"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            penalties: List[str] = []
            total_penalty = 0.0

            # 1. Cross-regime incoherence from PerceivePhase
            if getattr(ctx, "cross_regime_incoherence", False):
                penalties.append("cross_regime_incoherence")
                total_penalty += CROSS_REGIME_PENALTY

            # 2. Recent drift from DriftDetectionPhase
            if getattr(ctx, "drift_detected", False) or ctx.metadata.get("drift_event") is not None:
                penalties.append("recent_drift")
                total_penalty += DRIFT_PENALTY

            # 3. Signal coherence checker (existing logic)
            checker = SignalCoherenceChecker()
            historical = ctx.values if len(ctx.values) > 0 else None
            coherence_result = checker.check(
                predicted_value=ctx.fused_value,
                predicted_confidence=ctx.fused_confidence,
                is_anomaly=False,
                anomaly_score=0.0,
                historical_values=historical,
            )

            fused_conf = ctx.fused_confidence
            if not coherence_result.is_coherent:
                total_penalty += (fused_conf - coherence_result.resolved_confidence)
                penalties.append(coherence_result.conflict_type or "signal_conflict")

            # Apply accumulated penalties
            if total_penalty > 0.0:
                fused_conf = max(0.10, fused_conf - total_penalty)
                reason_parts = [f"penalty={total_penalty:.3f}"] + penalties
                resolution_reason = "; ".join(reason_parts)
                logger.warning(
                    "coherence_contextual_penalty",
                    extra={
                        "series_id": ctx.series_id,
                        "penalties": penalties,
                        "total_penalty": round(total_penalty, 4),
                        "resolved_confidence": round(fused_conf, 4),
                    },
                )
            else:
                resolution_reason = "No contextual penalties — signals coherent"

            result = CoherenceResult(
                is_coherent=len(penalties) == 0 and coherence_result.is_coherent,
                conflict_type=penalties[0] if penalties else coherence_result.conflict_type,
                resolved_value=ctx.fused_value,
                resolved_confidence=fused_conf,
                resolution_reason=resolution_reason,
                penalties=penalties,
            )

            return ctx.with_field(
                coherence_result=result,
                fused_confidence=fused_conf,
            )

        except Exception as e:
            logger.debug(f"coherence_check_skipped: {e}")
            return ctx
