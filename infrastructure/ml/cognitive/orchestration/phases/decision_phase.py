"""Decision Phase — Contextual decision engine integration.

Integrates ContextualDecisionEngine for adaptive strategy selection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from . import PipelineContext

try:
    from ...decision import ContextualDecisionEngine
except (ImportError, ModuleNotFoundError):
    ContextualDecisionEngine = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class DecisionPhase:
    """Phase: Contextual decision engine.
    
    Uses ContextualDecisionEngine for adaptive strategy selection,
    cost optimization, and context-aware decision making.
    """
    
    def __init__(self, decision_engine: Optional[Any] = None) -> None:
        """Initialize decision phase.
        
        Args:
            decision_engine: Optional ContextualDecisionEngine instance.
        """
        self._decision_engine = decision_engine
    
    @property
    def name(self) -> str:
        return "decision"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute decision phase.
        
        Args:
            ctx: Pipeline context with cognitive metrics.
        
        Returns:
            Updated context with decision strategy results.
        """
        # Skip if no decision engine available
        if self._decision_engine is None or ctx.decision_engine is None:
            return ctx
        
        try:
            # Use decision engine from context if available
            decision_engine = ctx.decision_engine if ctx.decision_engine else self._decision_engine
            
            # Prepare context for decision engine
            decision_context = {
                "series_id": ctx.series_id,
                "regime": ctx.regime,
                "confidence": ctx.fused_confidence or 0.0,
                "n_engines": len(ctx.perceptions) if ctx.perceptions else 0,
                "selected_engine": ctx.selected_engine,
                "is_fallback": ctx.is_fallback,
                "drift_detected": ctx.metadata.get("drift_detected", False),
            }
            
            try:
                # Run decision analysis
                decision_result = decision_engine.make_decision(
                    context=decision_context,
                    available_strategies=["aggressive", "conservative", "cost_optimized"],
                )
                
                # Extract decision strategy
                selected_strategy = getattr(decision_result, 'strategy', 'conservative')
                decision_confidence = getattr(decision_result, 'confidence', 0.0)
                cost_estimate = getattr(decision_result, 'cost_estimate', 0.0)
                
                # Log decision summary
                logger.debug(
                    "decision_analysis_completed",
                    extra={
                        "series_id": ctx.series_id,
                        "selected_strategy": selected_strategy,
                        "decision_confidence": decision_confidence,
                        "cost_estimate": cost_estimate,
                    },
                )
                
                return ctx.with_field(
                    decision_result=decision_result,
                    selected_strategy=selected_strategy,
                    decision_confidence=decision_confidence,
                    cost_estimate=cost_estimate,
                )
            except Exception as e:
                logger.debug(f"decision_analysis_failed: {e}")
        
        except Exception as e:
            logger.debug(f"decision_phase_failed: {e}")
        
        return ctx
