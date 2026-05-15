"""Inhibit Phase — MED-1 Refactoring.

Engine inhibition and weight mediation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

from core.parameters.numerical_constants import EPSILON

if TYPE_CHECKING:
    from . import PipelineContext

logger = logging.getLogger(__name__)


def _compute_signal_z_score(values: List[float]) -> float:
    """Compute z-score of the most recent value relative to the window."""
    if len(values) < 3:
        return 0.0
    
    historical = values[:-1]
    mean = sum(historical) / len(historical)
    variance = sum((x - mean) ** 2 for x in historical) / len(historical)
    std = variance ** 0.5
    
    if std < EPSILON.DIVISION:
        return 0.0
    
    return (values[-1] - mean) / std


class InhibitPhase:
    """Phase 4: Inhibition gate and weight mediation."""
    
    @property
    def name(self) -> str:
        return "inhibit"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute inhibition phase."""
        orchestrator = ctx.orchestrator
        
        # Compute signal z-score for anomaly override (CRIT-2)
        signal_z_score = _compute_signal_z_score(ctx.values) if ctx.values else 0.0
        
        # Compute inhibition states
        # Fallback to equal weights if plasticity_weights not available yet
        weights = ctx.plasticity_weights or {p.engine_name: 1.0 / len(ctx.perceptions) for p in (ctx.perceptions or [])}
        inh_states = orchestrator._inhibition.compute(
            ctx.perceptions,
            weights,
            ctx.error_dict,
            series_id=ctx.series_id,
            signal_z_score=signal_z_score,
        )
        
        # CRIT-2 FIX: Use resolved weights directly instead of non-existent _weight_mediator
        # Plasticity weights are already resolved by AdaptPhase before InhibitPhase executes.
        # The inhibition states are computed but we don't re-mediate weights (that was
        # the old WeightMediator pattern which was replaced by WeightResolutionService).
        # Instead, we use the plasticity_weights directly as the mediated_weights.
        mediated_weights = ctx.plasticity_weights
        
        # Update explanation builder
        if ctx.explanation and hasattr(ctx.explanation, 'set_inhibition'):
            ctx.explanation.set_inhibition(inh_states, mediated_weights)
        
        return ctx.with_field(
            inhibition_states=inh_states,
            mediated_weights=mediated_weights,
        )
