"""Assembly Phase — MED-1 Refactoring.

Constructs final PredictionResult from accumulated context.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from . import PipelineContext

from ....interfaces import PredictionResult

logger = logging.getLogger(__name__)


class AssemblyPhase:
    """Final phase: Assemble metadata and create PredictionResult."""
    
    @property
    def name(self) -> str:
        return "assembly"
    
    def execute(self, ctx: PipelineContext) -> PredictionResult:
        """Assemble final result from context."""
        metadata = self._build_metadata(ctx)
        
        # Get confidence interval if storage available
        orchestrator = ctx.orchestrator
        if orchestrator._storage and ctx.series_id != "unknown":
            try:
                ci = orchestrator._storage.compute_confidence_interval(
                    ctx.series_id, ctx.selected_engine, ctx.fused_value
                )
                if ci:
                    metadata["confidence_interval"] = ci
            except Exception as e:
                logger.debug(f"confidence_interval_failed: {e}")
        
        return PredictionResult(
            predicted_value=ctx.fused_value,
            confidence=ctx.fused_confidence,
            trend=ctx.fused_trend,
            metadata=metadata,
        )
    
    def _build_metadata(self, ctx: PipelineContext) -> Dict[str, Any]:
        """Build comprehensive metadata dict."""
        metadata: Dict[str, Any] = {
            "cognitive_diagnostic": ctx.diagnostic.to_dict() if ctx.diagnostic else None,
            "explanation": ctx.explanation.to_dict() if ctx.explanation else None,
            "pipeline_timing": ctx.timer.to_dict() if ctx.timer else None,
        }
        
        # Add boundary check if available
        if ctx.boundary_result is not None and ctx.boundary_result.within_domain:
            metadata["boundary_check"] = {
                "within_domain": True,
                "data_quality_score": ctx.boundary_result.data_quality_score,
                "warnings": ctx.boundary_result.warnings,
            }
        
        # Add coherence check
        if ctx.coherence_result is not None:
            metadata["coherence_check"] = {
                "is_coherent": ctx.coherence_result.is_coherent,
                "conflict_type": ctx.coherence_result.conflict_type,
                "resolved_confidence": ctx.coherence_result.resolved_confidence,
                "resolution_reason": ctx.coherence_result.resolution_reason,
            }
        
        # Add engine decision
        if ctx.engine_decision is not None:
            metadata["engine_decision"] = {
                "chosen_engine": ctx.engine_decision.chosen_engine,
                "authority": ctx.engine_decision.authority,
                "reason": ctx.engine_decision.reason,
                "overrides": ctx.engine_decision.overrides,
            }
        
        # Add calibration report
        if ctx.calibrated_confidence is not None:
            metadata["calibration_report"] = {
                "raw": ctx.calibrated_confidence.raw,
                "calibrated": ctx.calibrated_confidence.calibrated,
                "penalty_applied": ctx.calibrated_confidence.penalty_applied,
                "reasons": ctx.calibrated_confidence.reasons,
            }
        
        # Add action guard
        if ctx.guarded_action is not None:
            metadata["action_guard"] = {
                "action_allowed": ctx.guarded_action.action_allowed,
                "original_action": ctx.guarded_action.original_action,
                "final_action": ctx.guarded_action.final_action,
                "suppressed_reason": ctx.guarded_action.suppressed_reason,
                "series_state": ctx.guarded_action.series_state,
            }
        
        # Add unified narrative
        if ctx.unified_narrative is not None:
            metadata["unified_narrative"] = {
                "primary_verdict": ctx.unified_narrative.primary_verdict,
                "severity": ctx.unified_narrative.severity,
                "confidence": ctx.unified_narrative.confidence,
                "contradictions": ctx.unified_narrative.contradictions,
                "sources_used": ctx.unified_narrative.sources_used,
                "suppressed": ctx.unified_narrative.suppressed,
            }
        
        return metadata
