"""Inhibit Phase — MED-1 Refactoring.

Engine inhibition and weight mediation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

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
    
    if std < 1e-9:
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
        inh_states = orchestrator._inhibition.compute(
            ctx.perceptions,
            ctx.plasticity_weights,
            ctx.error_dict,
            series_id=ctx.series_id,
            signal_z_score=signal_z_score,
        )
        
        # Mediate weights
        mediated_weights = orchestrator._weight_mediator.mediate(
            ctx.plasticity_weights, inh_states
        )
        
        # Update explanation builder
        if ctx.explanation and hasattr(ctx.explanation, 'set_inhibition'):
            ctx.explanation.set_inhibition(inh_states, mediated_weights)
        
        return ctx.with_field(
            inhibition_states=inh_states,
            mediated_weights=mediated_weights,
        )
