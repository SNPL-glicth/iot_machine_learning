"""Coherence Check Phase — MED-1 Refactoring.

Signal coherence validation when enabled.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import PipelineContext

from ......domain.services.signal_coherence_checker import SignalCoherenceChecker

logger = logging.getLogger(__name__)


class CoherenceCheckPhase:
    """Phase 7: Signal coherence check (optional)."""
    
    @property
    def name(self) -> str:
        return "coherence_check"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute coherence check if enabled."""
        flags = ctx.flags
        
        if not flags.ML_COHERENCE_CHECK_ENABLED:
            return ctx
        
        try:
            checker = SignalCoherenceChecker()
            historical = ctx.values if len(ctx.values) > 0 else None
            
            coherence_result = checker.check(
                predicted_value=ctx.fused_value,
                predicted_confidence=ctx.fused_confidence,
                is_anomaly=False,  # Placeholder
                anomaly_score=0.0,
                historical_values=historical,
            )
            
            fused_conf = ctx.fused_confidence
            if not coherence_result.is_coherent:
                fused_conf = coherence_result.resolved_confidence
                logger.warning("coherence_conflict_detected", extra={
                    "conflict_type": coherence_result.conflict_type,
                    "resolved_confidence": round(fused_conf, 4),
                    "reason": coherence_result.resolution_reason,
                })
            
            return ctx.with_field(
                coherence_result=coherence_result,
                fused_confidence=fused_conf,
            )
            
        except Exception as e:
            logger.debug(f"coherence_check_skipped: {e}")
            return ctx
