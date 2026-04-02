"""Decision Arbiter Phase — MED-1 Refactoring.

Engine decision arbitration when enabled.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import PipelineContext

from ......domain.services.engine_decision_arbiter import EngineDecisionArbiter

logger = logging.getLogger(__name__)


class DecisionArbiterPhase:
    """Phase 6: Engine decision arbitration (optional)."""
    
    @property
    def name(self) -> str:
        return "decision_arbiter"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute decision arbiter if enabled."""
        flags = ctx.flags
        
        if not flags.ML_DECISION_ARBITER_ENABLED:
            return ctx
        
        try:
            flag_engine = (
                flags.get_active_engine_for_series(ctx.series_id) 
                if hasattr(flags, 'get_active_engine_for_series') else "unknown"
            )
            profile_engine = ctx.selected_engine
            fusion_engine = ctx.selected_engine
            
            arbiter = EngineDecisionArbiter()
            engine_decision = arbiter.arbitrate(
                flag_engine=flag_engine,
                profile_engine=profile_engine,
                fusion_engine=fusion_engine,
                series_id=ctx.series_id,
                rollback_to_baseline=flags.ML_ROLLBACK_TO_BASELINE,
                series_overrides=getattr(flags, 'ML_ENGINE_SERIES_OVERRIDES', {}),
            )
            
            logger.info("engine_decision_arbitrated", extra={
                "series_id": ctx.series_id,
                "chosen_engine": engine_decision.chosen_engine,
                "authority": engine_decision.authority,
                "reason": engine_decision.reason,
                "overrides": engine_decision.overrides,
            })
            
            return ctx.with_field(engine_decision=engine_decision)
            
        except Exception as e:
            logger.debug(f"engine_decision_arbiter_skipped: {e}")
            return ctx
