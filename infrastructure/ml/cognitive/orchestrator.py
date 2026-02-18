"""Meta-Cognitive Orchestrator — the brain of UTSAE.

Pipeline: Perceive → Predict → Inhibit → Adapt → Fuse → Explain.
Delegates computation to sub-modules.  No persistence, no I/O beyond logging.
"""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional

from ..interfaces import PredictionEngine, PredictionResult
from .engine_selector import WeightedFusion
from .inhibition import InhibitionConfig, InhibitionGate
from .plasticity import PlasticityTracker
from .signal_analyzer import SignalAnalyzer
from .explanation_builder import ExplanationBuilder
from .types import EnginePerception, MetaDiagnostic, PipelineTimer
from .orchestrator_helpers import collect_perceptions, create_fallback_result
from .plasticity_factory import build_advanced_plasticity, null_advanced_plasticity
from .record_actual_handler import record_actual_dispatch

from iot_machine_learning.domain.entities.series.structural_analysis import (
    StructuralAnalysis,
)

logger = logging.getLogger(__name__)
_MAX_ERROR_HISTORY: int = 50


class MetaCognitiveOrchestrator(PredictionEngine):
    """Orchestrates multiple engines with cognitive reasoning.

    Implements PredictionEngine — drop-in replacement for any single engine.
    """

    def __init__(
        self,
        engines: List[PredictionEngine],
        initial_weights: Optional[Dict[str, float]] = None,
        inhibition_config: Optional[InhibitionConfig] = None,
        enable_plasticity: bool = True,
        budget_ms: float = 500.0,
        storage_adapter=None,  # type: ignore[assignment]
        enable_advanced_plasticity: bool = False,  # FASE 7: Advanced plasticity flag
    ) -> None:
        if not engines:
            raise ValueError("At least one engine required")
        self._engines = engines
        self._analyzer = SignalAnalyzer()
        self._inhibition = InhibitionGate(inhibition_config)
        self._fusion = WeightedFusion()
        self._plasticity = PlasticityTracker() if enable_plasticity else None
        if initial_weights:
            self._base_weights = dict(initial_weights)
        else:
            n = len(engines)
            self._base_weights = {e.name: 1.0 / n for e in engines}
        self._recent_errors: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=_MAX_ERROR_HISTORY)
        )
        self._last_diagnostic: Optional[MetaDiagnostic] = None
        self._last_explanation = None  # type: ignore[assignment]
        # Internal state for record_actual (avoids reading back from MetaDiagnostic)
        self._last_regime: Optional[str] = None
        self._last_perceptions: List[EnginePerception] = []
        self._budget_ms = budget_ms
        # FASE 3: Storage adapter for adaptive weights (optional)
        self._storage = storage_adapter
        self._adaptive_epsilon = 0.01  # Prevent division by zero
        self._last_timer: Optional[PipelineTimer] = None
        
        # Weight adjustment service
        from .weight_adjustment_service import WeightAdjustmentService
        self._weight_service = WeightAdjustmentService(
            base_weights=self._base_weights,
            storage_adapter=storage_adapter,
            plasticity_tracker=self._plasticity,
            epsilon=self._adaptive_epsilon,
        )
        
        # FASE 7: Advanced Plasticity System
        self._enable_advanced_plasticity = enable_advanced_plasticity
        if enable_advanced_plasticity:
            (
                self._plasticity_coordinator,
                self._adaptive_lr,
                self._asymmetric_penalty,
                self._contextual_tracker,
                self._health_monitor,
            ) = build_advanced_plasticity(storage_adapter)
        else:
            (
                self._plasticity_coordinator,
                self._adaptive_lr,
                self._asymmetric_penalty,
                self._contextual_tracker,
                self._health_monitor,
            ) = null_advanced_plasticity()
        
        # Store last context for advanced plasticity
        self._last_plasticity_context = None

    @property
    def name(self) -> str:
        return "meta_cognitive_orchestrator"

    def can_handle(self, n_points: int) -> bool:
        return any(e.can_handle(n_points) for e in self._engines)

    def predict(
        self,
        values: List[float],
        timestamps: Optional[List[float]] = None,
        series_id: str = "unknown",
    ) -> PredictionResult:
        timer = PipelineTimer(budget_ms=self._budget_ms)

        # Phase: PERCEIVE
        timer.start()
        profile = self._analyzer.analyze(values, timestamps)
        regime_str = profile.regime.value
        builder = ExplanationBuilder(series_id)
        builder.set_signal(profile)
        timer.stop("perceive")
        
        # FASE 7: Create PlasticityContext for advanced plasticity
        if self._enable_advanced_plasticity and self._plasticity_coordinator:
            self._last_plasticity_context = self._plasticity_coordinator.create_plasticity_context(
                profile, series_id
            )
        else:
            self._last_plasticity_context = None

        # Phase: PREDICT
        timer.start()
        perceptions = collect_perceptions(self._engines, values, timestamps)
        timer.stop("predict")

        if not perceptions:
            self._last_timer = timer
            return self._handle_fallback(values, profile, builder, timer, "no_valid_perceptions")

        # Budget guard: if perceive+predict already over budget, cut to fallback
        if timer.total_ms > timer.budget_ms:
            logger.warning("pipeline_budget_exceeded", extra={
                "phase": "predict", "elapsed_ms": round(timer.total_ms, 2),
                "budget_ms": timer.budget_ms,
            })
            self._last_timer = timer
            return self._handle_fallback(values, profile, builder, timer, "budget_exceeded")

        builder.set_perceptions(perceptions, n_engines_total=len(self._engines))

        # Phase: ADAPT
        timer.start()
        adapted = (self._plasticity is not None
                   and self._plasticity.has_history(regime_str))
        weights = self._weight_service.resolve_weights(regime_str, [p.engine_name for p in perceptions], series_id=series_id)
        builder.set_adaptation(adapted=adapted, regime=regime_str)
        timer.stop("adapt")

        # Phase: INHIBIT
        timer.start()
        error_dict = {k: list(v) for k, v in self._recent_errors.items()}
        inh_states = self._inhibition.compute(perceptions, weights, error_dict)
        builder.set_inhibition(inh_states, weights)
        timer.stop("inhibit")

        # Phase: FUSE
        timer.start()
        (fused_val, fused_conf, fused_trend,
         final_weights, selected, reason) = self._fusion.fuse(
            perceptions, inh_states)

        method = "weighted_average" if len(perceptions) > 1 else "single_engine"
        builder.set_fusion(
            fused_val, fused_conf, fused_trend,
            final_weights, selected, reason, method,
        )
        timer.stop("fuse")

        # Phase: EXPLAIN
        timer.start()
        diag = MetaDiagnostic(
            signal_profile=profile, perceptions=perceptions,
            inhibition_states=inh_states, final_weights=final_weights,
            selected_engine=selected, selection_reason=reason,
            fusion_method=method,
        )
        self._last_diagnostic = diag
        self._last_explanation = builder.build()
        timer.stop("explain")

        self._last_regime = regime_str
        self._last_perceptions = list(perceptions)
        self._last_timer = timer

        logger.debug("cognitive_prediction", extra={
            "n_engines": len(perceptions), "selected": selected,
            "regime": regime_str, "fused_value": round(fused_val, 4),
            "pipeline_ms": round(timer.total_ms, 2),
        })
        if timer.is_over_budget:
            logger.warning("pipeline_over_budget", extra=timer.to_dict())

        # FASE 4: Add confidence interval to metadata (optional, non-breaking)
        metadata = {
            "cognitive_diagnostic": diag.to_dict(),
            "explanation": self._last_explanation.to_dict(),
            "pipeline_timing": timer.to_dict(),
        }
        
        if self._storage and series_id != "unknown":
            ci = self._storage.compute_confidence_interval(
                series_id, selected, fused_val
            )
            if ci:
                metadata["confidence_interval"] = ci

        return PredictionResult(
            predicted_value=fused_val, confidence=fused_conf,
            trend=fused_trend,
            metadata=metadata,
        )

    def record_actual(
        self,
        actual_value: float,
        series_id: Optional[str] = None,
        series_context=None,  # type: ignore[assignment]  # SeriesContext for asymmetric penalty
    ) -> None:
        """Record true value for plasticity learning and error tracking.

        FASE 1: Also records prediction errors to database for adaptive learning.
        FASE 7: Integrates advanced plasticity system when enabled.
        """
        record_actual_dispatch(
            actual_value=actual_value,
            last_regime=self._last_regime,
            last_perceptions=self._last_perceptions,
            last_plasticity_context=self._last_plasticity_context,
            enable_advanced_plasticity=self._enable_advanced_plasticity,
            plasticity_coordinator=self._plasticity_coordinator,
            plasticity_tracker=self._plasticity,
            recent_errors=self._recent_errors,
            storage=self._storage,
            series_id=series_id,
            series_context=series_context,
        )
    

    @property
    def last_diagnostic(self) -> Optional[MetaDiagnostic]:
        """Full reasoning trace.

        .. deprecated:: 2.0
            Use ``last_explanation`` instead, which returns a domain-pure
            ``Explanation`` value object.  ``MetaDiagnostic`` will be
            removed in a future version.
        """
        warnings.warn(
            "last_diagnostic is deprecated. Use last_explanation instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._last_diagnostic

    @property
    def last_explanation(self):
        """Last Explanation produced (domain value object)."""
        return self._last_explanation

    @property
    def last_pipeline_timing(self) -> Optional[PipelineTimer]:
        """Per-phase latency of the last predict() call."""
        return self._last_timer
    
    def _handle_fallback(
        self, values: List[float], profile: StructuralAnalysis,
        builder: ExplanationBuilder, timer: PipelineTimer, reason: str
    ) -> PredictionResult:
        """Handle fallback case and update internal state."""
        result = create_fallback_result(values, profile, builder, timer, reason)
        self._last_diagnostic = MetaDiagnostic(
            signal_profile=profile, perceptions=[], inhibition_states=[],
            final_weights={}, selected_engine="none",
            selection_reason="all_engines_failed",
            fusion_method="fallback", fallback_reason=reason,
        )
        self._last_explanation = builder.build()
        self._last_regime = profile.regime.value
        self._last_perceptions = []
        return result
