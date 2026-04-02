"""Action Guard Phase — MED-1 Refactoring.

Action suppression based on series state when enabled.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import PipelineContext

from ......domain.services.action_guard import ActionGuard

logger = logging.getLogger(__name__)


class ActionGuardPhase:
    """Phase 10: Action guard (optional, last phase)."""
    
    @property
    def name(self) -> str:
        return "action_guard"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute action guard if enabled."""
        flags = ctx.flags
        
        if not flags.ML_ACTION_GUARD_ENABLED:
            return ctx
        
        try:
            series_state = getattr(ctx.profile, 'series_state', 'UNKNOWN')
            
            # Extract action info from explanation
            action_required = False
            recommended_action = None
            severity = "NORMAL"
            
            if ctx.explanation:
                outcome = getattr(ctx.explanation, 'outcome', None)
                if outcome:
                    severity = getattr(outcome, 'severity', 'NORMAL')
                    action_required = getattr(outcome, 'action_required', False)
                    recommended_action = getattr(outcome, 'recommended_action', None)
            
            guard = ActionGuard()
            guarded_action = guard.guard(
                action_required=action_required,
                recommended_action=recommended_action,
                severity=severity,
                series_state=series_state,
            )
            
            if not guarded_action.action_allowed:
                logger.warning("action_suppressed", extra={
                    "series_id": ctx.series_id,
                    "series_state": series_state,
                    "original_action": recommended_action,
                    "reason": guarded_action.suppressed_reason,
                })
            
            return ctx.with_field(guarded_action=guarded_action)
            
        except Exception as e:
            logger.debug(f"action_guard_skipped: {e}")
            return ctx
