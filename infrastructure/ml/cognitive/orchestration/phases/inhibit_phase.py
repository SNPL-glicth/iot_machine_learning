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

        # Fase 3: período de gracia post-evento (suprimir inhibición agresiva)
        feature_ctx = getattr(ctx, "feature_context", None)
        event_ctx = getattr(feature_ctx, "event_context", None) if feature_ctx else None

        in_stabilization = event_ctx is not None and event_ctx.is_active
        if in_stabilization:
            logger.debug(
                f"inhibit_stabilization_gate_active series={ctx.series_id} "
                f"event={event_ctx.detected_event.value}"
            )

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

        # Durante estabilización: reducir supresión al 50%
        if in_stabilization:
            from ...analysis.types import InhibitionState
            inh_states = [
                InhibitionState(
                    engine_name=s.engine_name,
                    base_weight=s.base_weight,
                    inhibited_weight=s.base_weight * 0.5 + s.inhibited_weight * 0.5,
                    inhibition_reason=f"stabilization_gate:{s.inhibition_reason}",
                    suppression_factor=s.suppression_factor * 0.5,
                )
                for s in inh_states
            ]

        # CRIT-2 FIX: Use resolved weights directly instead of non-existent _weight_mediator
        mediated_weights = ctx.plasticity_weights

        # Update explanation builder
        if ctx.explanation and hasattr(ctx.explanation, "set_inhibition"):
            ctx.explanation.set_inhibition(inh_states, mediated_weights)

        return ctx.with_field(
            inhibition_states=inh_states,
            mediated_weights=mediated_weights,
        )
