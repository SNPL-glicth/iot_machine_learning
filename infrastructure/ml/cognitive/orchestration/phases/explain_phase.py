"""Explain Phase — Causal Narrative Refactor.

Generates human-readable explanations linking metrics to causes.
Maps technical indicators to intuitive reasoning narratives.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from . import PipelineContext

from ...analysis.types import MetaDiagnostic

logger = logging.getLogger(__name__)


class CausalNarrativeBuilder:
    """Builds causal explanations from signal metrics.
    
    Maps technical indicators to human-readable reasoning.
    """
    
    @staticmethod
    def from_signal_profile(profile) -> List[str]:
        """Generate causal narratives from signal profile."""
        narratives = []
        
        if profile is None:
            return narratives
        
        # Anomaly detection narrative
        z_score = getattr(profile, 'z_score', None)
        if z_score is not None and z_score > 2.5:
            narratives.append(
                "Detección de anomalía por cambio súbito de magnitud"
            )
        
        # Stability narrative
        stability = getattr(profile, 'stability', None)
        if stability is not None and stability < 0.3:
            narratives.append(
                "Predicción conservadora debido a alta inestabilidad en la señal"
            )
        
        # Regime narrative
        regime = getattr(profile, 'regime', None)
        if regime == "VOLATILE":
            narratives.append(
                "Alta volatilidad detectada: adaptando pesos dinámicamente"
            )
        elif regime == "TRENDING":
            trend_dir = getattr(profile, 'trend_direction', 'up')
            narratives.append(
                f"Tendencia {trend_dir} establecida: extrapolando momentum"
            )
        
        # Noise narrative
        noise_ratio = getattr(profile, 'noise_ratio', None)
        if noise_ratio is not None and noise_ratio > 0.3:
            narratives.append(
                "Señal con ruido significativo: aplicando filtrado adaptativo"
            )
        
        return narratives
    
    @staticmethod
    def from_perceptions(perceptions) -> List[str]:
        """Generate narratives from engine perceptions."""
        narratives = []
        
        if not perceptions:
            return narratives
        
        inhibited = [p for p in perceptions if getattr(p, 'inhibited', False)]
        active = [p for p in perceptions if not getattr(p, 'inhibited', False)]
        
        if len(inhibited) > len(active):
            narratives.append(
                f"Mayoría de engines inhibidos ({len(inhibited)}/{len(perceptions)}): "
                "usando fallback conservador"
            )
        
        if len(active) == 1:
            narratives.append(
                f"Engine único activo: {active[0].engine_name}"
            )
        
        return narratives


class ExplainPhase:
    """Phase 9: Explanation and diagnostic generation."""
    
    @property
    def name(self) -> str:
        return "explain"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute explanation phase with causal narratives."""
        orchestrator = ctx.orchestrator
        
        # Generate causal narratives
        signal_narratives = CausalNarrativeBuilder.from_signal_profile(ctx.profile)
        perception_narratives = CausalNarrativeBuilder.from_perceptions(ctx.perceptions)
        
        # Combine all narratives
        all_narratives = signal_narratives + perception_narratives
        
        # Create meta diagnostic with narratives
        diag = MetaDiagnostic(
            signal_profile=ctx.profile,
            perceptions=ctx.perceptions,
            inhibition_states=ctx.inhibition_states,
            final_weights=ctx.final_weights,
            selected_engine=ctx.selected_engine,
            selection_reason=ctx.selection_reason,
            fusion_method=ctx.fusion_method,
        )
        
        # Build explanation with narratives
        explanation_dict = {
            "selected_engine": ctx.selected_engine,
            "selection_reason": ctx.selection_reason,
            "regime": ctx.regime,
            "narratives": all_narratives,
            "n_engines_active": len([p for p in (ctx.perceptions or []) 
                                    if not getattr(p, 'inhibited', False)]),
            "n_engines_inhibited": len([p for p in (ctx.perceptions or []) 
                                       if getattr(p, 'inhibited', False)]),
        }
        
        # Update orchestrator via state_manager (R-1 compatible)
        if hasattr(orchestrator, 'update_series_state'):
            orchestrator.update_series_state(
                series_id=ctx.series_id,
                regime=ctx.regime,
                perceptions=list(ctx.perceptions) if ctx.perceptions else [],
            )
        
        # Store diagnostic and explanation
        orchestrator._last_diagnostic = diag
        orchestrator._last_explanation = explanation_dict
        orchestrator._last_timer = ctx.timer
        
        # Log with narratives
        logger.debug("cognitive_prediction", extra={
            "n_engines": len(ctx.perceptions) if ctx.perceptions else 0,
            "selected": ctx.selected_engine,
            "regime": ctx.regime,
            "fused_value": round(ctx.fused_value, 4) if ctx.fused_value else None,
            "pipeline_ms": round(ctx.timer.total_ms, 2),
            "narratives": all_narratives,
        })
        
        if ctx.timer.is_over_budget:
            logger.warning("pipeline_over_budget", extra=ctx.timer.to_dict())
        
        return ctx.with_field(
            diagnostic=diag,
            explanation=explanation_dict,
        )
