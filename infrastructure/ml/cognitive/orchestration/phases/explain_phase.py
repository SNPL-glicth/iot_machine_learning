"""Explain Phase — MED-1 Refactoring.

Meta-diagnostic generation and explanation building.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import PipelineContext

from ...analysis.types import MetaDiagnostic

logger = logging.getLogger(__name__)


class ExplainPhase:
    """Phase 9: Explanation and diagnostic generation."""
    
    @property
    def name(self) -> str:
        return "explain"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute explanation phase."""
        orchestrator = ctx.orchestrator
        
        # Create meta diagnostic
        diag = MetaDiagnostic(
            signal_profile=ctx.profile,
            perceptions=ctx.perceptions,
            inhibition_states=ctx.inhibition_states,
            final_weights=ctx.final_weights,
            selected_engine=ctx.selected_engine,
            selection_reason=ctx.selection_reason,
            fusion_method=ctx.fusion_method,
        )
        
        # Build explanation
        if ctx.explanation and hasattr(ctx.explanation, 'build'):
            explanation = ctx.explanation.build()
        else:
            explanation = None
        
        # Update orchestrator state
        orchestrator._last_diagnostic = diag
        orchestrator._last_explanation = explanation
        orchestrator._last_regime = ctx.regime
        orchestrator._last_perceptions = list(ctx.perceptions) if ctx.perceptions else []
        orchestrator._last_timer = ctx.timer
        
        # Log prediction
        logger.debug("cognitive_prediction", extra={
            "n_engines": len(ctx.perceptions) if ctx.perceptions else 0,
            "selected": ctx.selected_engine,
            "regime": ctx.regime,
            "fused_value": round(ctx.fused_value, 4) if ctx.fused_value else None,
            "pipeline_ms": round(ctx.timer.total_ms, 2),
        })
        
        if ctx.timer.is_over_budget:
            logger.warning("pipeline_over_budget", extra=ctx.timer.to_dict())
        
        return ctx.with_field(
            diagnostic=diag,
            explanation=explanation,
        )
