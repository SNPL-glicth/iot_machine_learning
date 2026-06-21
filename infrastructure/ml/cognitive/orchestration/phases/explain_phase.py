"""Explain Phase — Causal Narrative Refactor.

Generates human-readable explanations linking metrics to causes.
Maps technical indicators to intuitive reasoning narratives.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

from .context import PipelineContext
from iot_machine_learning.application.explainability.explanation_renderer import ExplanationRenderer

from ...analysis.types import MetaDiagnostic

try:
    from ...observability import ExplainabilityValidator
except (ImportError, ModuleNotFoundError):
    ExplainabilityValidator = None  # type: ignore[assignment,misc]

try:
    from ...explainability import ContextualExplainabilityEngine
except (ImportError, ModuleNotFoundError):
    ContextualExplainabilityEngine = None  # type: ignore[assignment,misc]

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
    
    def __init__(
        self,
        explainability_validator: Optional[Any] = None,
        contextual_explainability_engine: Optional[Any] = None,
    ) -> None:
        """Initialize explain phase.
        
        Args:
            explainability_validator: Optional ExplainabilityValidator instance.
            contextual_explainability_engine: Optional ContextualExplainabilityEngine instance.
        """
        self._explainability_validator = explainability_validator
        if ExplainabilityValidator is not None and self._explainability_validator is None:
            self._explainability_validator = ExplainabilityValidator()
        
        self._contextual_explainability_engine = contextual_explainability_engine
        if ContextualExplainabilityEngine is not None and self._contextual_explainability_engine is None:
            self._contextual_explainability_engine = ContextualExplainabilityEngine()
    
    @property
    def name(self) -> str:
        return "explain"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute explanation phase."""
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
        
        # Store diagnostic and explanation (thread-safe)
        with orchestrator._state_lock:
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
        
        # Generate human-readable summary from explanation dict
        explanation_summary = None
        if explanation_dict is not None:
            parts = []
            if explanation_dict.get("method"):
                parts.append(f"Método: {explanation_dict['method']}")
            if explanation_dict.get("regime"):
                parts.append(f"Régimen: {explanation_dict['regime']}")
            if explanation_dict.get("influencing_factors"):
                factors = explanation_dict["influencing_factors"]
                parts.append(f"Factores ({len(factors)}): {', '.join(str(f) for f in factors[:3])}")
            if explanation_dict.get("recommendations_for_inspector"):
                recs = explanation_dict["recommendations_for_inspector"]
                parts.append(f"Recomendaciones ({len(recs)})")
            if parts:
                explanation_summary = " | ".join(parts)

        # RUL enrichment
        from iot_machine_learning.infrastructure.ml.anomaly.rul import (
            RULEstimator, RULNarrator
        )

        _REGIME_DRIFT_MAP = {
            "STABLE": 0.0, "TRENDING": 0.3,
            "NOISY": 0.4, "VOLATILE": 0.6,
            "TRANSITIONAL": 0.5,
        }

        try:
            rul_score = min(
                abs(getattr(ctx.profile, "z_score", 0.0)) / 4.0,
                1.0,
            )
            rul_drift = _REGIME_DRIFT_MAP.get(
                getattr(ctx, "regime", "STABLE"), 0.0
            )
            rul_consecutive = getattr(ctx, "consecutive_anomalies", 0)

            rul_estimate = RULEstimator().estimate(
                anomaly_score=rul_score,
                drift_magnitude=rul_drift,
                consecutive_anomalies=rul_consecutive,
            )
            rul_narrative = RULNarrator().narrate(rul_estimate)

            if rul_narrative:
                all_narratives.append(rul_narrative)
        except Exception:
            pass  # RUL never breaks the pipeline

        # Record explainability metrics (Phase 3C)
        if ctx.metrics_collector is not None:
            try:
                ctx.metrics_collector.record_explainability(
                    explanation_quality=len(all_narratives),
                    explanation_coverage=1.0 if all_narratives else 0.0,
                )
            except Exception as e:
                logger.debug(f"metrics_collection_failed: {e}")
        
        # Validate explainability quality (Phase 3C)
        validation_result = None
        if self._explainability_validator is not None:
            try:
                # Create a simple explanation object for validation
                from domain.entities.explainability import ContextualExplanation
                simple_explanation = ContextualExplanation(
                    sensor_id=ctx.series_id,
                    sensor_type="generic",
                    current_regime=ctx.regime or "unknown",
                    anomaly_score=getattr(ctx.profile, "z_score", 0.0) if ctx.profile else 0.0,
                    operational_confidence=ctx.fused_confidence or 0.0,
                    primary_drivers=[ctx.selected_engine] if ctx.selected_engine else [],
                    suggested_actions=all_narratives[:3],
                    timestamp=time.time(),
                )
                
                validation_result = self._explainability_validator.validate_explanation(
                    explanation=simple_explanation,
                    retrieval_relevance=ctx.fused_confidence or 0.0,
                )
                
                logger.debug(
                    "explainability_validation",
                    extra={
                        "series_id": ctx.series_id,
                        "quality_score": validation_result.get("explainability_quality_score", 0.0),
                    },
                )
            except Exception as e:
                logger.debug(f"explainability_validation_failed: {e}")
        
        # Generate contextual explanation (Phase 3B)
        contextual_explanation = None
        if self._contextual_explainability_engine is not None and ctx.memory_registry is not None:
            try:
                contextual_explanation = self._contextual_explainability_engine.generate_explanation(
                    sensor_id=ctx.series_id,
                    regime=ctx.regime or "unknown",
                    anomaly_score=getattr(ctx.profile, "z_score", 0.0) if ctx.profile else 0.0,
                    confidence=ctx.fused_confidence or 0.0,
                    memory_registry=ctx.memory_registry,
                )
                
                if contextual_explanation:
                    # Add contextual explanation to narratives
                    all_narratives.extend(contextual_explanation.get("narratives", []))
                    logger.debug(
                        "contextual_explanation_generated",
                        extra={
                            "series_id": ctx.series_id,
                            "contextual_confidence": contextual_explanation.get("confidence", 0.0),
                        },
                    )
            except Exception as e:
                logger.debug(f"contextual_explanation_failed: {e}")

        return ctx.with_field(
            diagnostic=diag,
            explanation=explanation_dict,
            explanation_summary=explanation_summary,
            validation_result=validation_result,
            contextual_explanation=contextual_explanation,
        )
