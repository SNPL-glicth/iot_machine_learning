"""Narrative Unification Phase — MED-1 Refactoring.

Unifies narratives from multiple sources when enabled.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import PipelineContext

from ......domain.services.narrative_unifier import NarrativeUnifier

logger = logging.getLogger(__name__)


class NarrativeUnificationPhase:
    """Phase 11: Narrative unification (optional, absolute last phase)."""
    
    @property
    def name(self) -> str:
        return "narrative_unification"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute narrative unification if enabled."""
        flags = ctx.flags
        
        if not flags.ML_NARRATIVE_UNIFICATION_ENABLED:
            return ctx
        
        try:
            unifier = NarrativeUnifier()
            
            # Build narrative sources
            prediction_explanation = None
            anomaly_narrative = None
            text_narrative = None
            
            if ctx.explanation:
                outcome = getattr(ctx.explanation, 'outcome', None)
                if outcome:
                    prediction_explanation = {
                        "verdict": getattr(outcome, 'description', None),
                        "severity": getattr(outcome, 'severity', 'UNKNOWN'),
                        "confidence": ctx.fused_confidence,
                    }
            
            # Use coherence result as anomaly proxy
            if ctx.coherence_result and not ctx.coherence_result.is_coherent:
                anomaly_narrative = {
                    "verdict": f"coherence_conflict:{ctx.coherence_result.conflict_type}",
                    "severity": "WARNING",
                    "confidence": ctx.coherence_result.resolved_confidence,
                }
            
            unified_narrative = unifier.unify(
                prediction_explanation=prediction_explanation,
                anomaly_narrative=anomaly_narrative,
                text_narrative=text_narrative,
            )
            
            if unified_narrative.contradictions:
                logger.warning("narrative_contradictions_detected", extra={
                    "series_id": ctx.series_id,
                    "contradictions": unified_narrative.contradictions,
                    "unified_severity": unified_narrative.severity,
                })
            
            return ctx.with_field(unified_narrative=unified_narrative)
            
        except Exception as e:
            logger.debug(f"narrative_unification_skipped: {e}")
            return ctx
