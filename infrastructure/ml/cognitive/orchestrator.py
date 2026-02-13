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
        self._last_timer: Optional[PipelineTimer] = None

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

        # Phase: PREDICT
        timer.start()
        perceptions = self._collect_perceptions(values, timestamps)
        timer.stop("predict")

        if not perceptions:
            self._last_timer = timer
            return self._fallback(values, profile, builder, timer)

        # Budget guard: if perceive+predict already over budget, cut to fallback
        if timer.total_ms > timer.budget_ms:
            logger.warning("pipeline_budget_exceeded", extra={
                "phase": "predict", "elapsed_ms": round(timer.total_ms, 2),
                "budget_ms": timer.budget_ms,
            })
            self._last_timer = timer
            return self._fallback(
                values, profile, builder, timer,
                reason="budget_exceeded",
            )

        builder.set_perceptions(perceptions, n_engines_total=len(self._engines))

        # Phase: ADAPT
        timer.start()
        adapted = (self._plasticity is not None
                   and self._plasticity.has_history(regime_str))
        weights = self._resolve_weights(regime_str, perceptions)
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

        return PredictionResult(
            predicted_value=fused_val, confidence=fused_conf,
            trend=fused_trend,
            metadata={
                "cognitive_diagnostic": diag.to_dict(),
                "explanation": self._last_explanation.to_dict(),
                "pipeline_timing": timer.to_dict(),
            },
        )

    def record_actual(self, actual_value: float) -> None:
        """Record true value for plasticity learning and error tracking."""
        if self._last_regime is None or not self._last_perceptions:
            return
        regime = self._last_regime
        for p in self._last_perceptions:
            error = abs(p.predicted_value - actual_value)
            self._recent_errors[p.engine_name].append(error)
            if self._plasticity is not None:
                self._plasticity.update(regime, p.engine_name, error)

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

    # -- private -----------------------------------------------------------

    def _collect_perceptions(
        self, values: List[float], ts: Optional[List[float]],
    ) -> List[EnginePerception]:
        out: List[EnginePerception] = []
        for eng in self._engines:
            if not eng.can_handle(len(values)):
                continue
            try:
                r = eng.predict(values, ts)
                d = r.metadata.get("diagnostic", {}) or {}
                out.append(EnginePerception(
                    engine_name=eng.name,
                    predicted_value=r.predicted_value,
                    confidence=r.confidence, trend=r.trend,
                    stability=d.get("stability_indicator", 0.0) if isinstance(d, dict) else 0.0,
                    local_fit_error=d.get("local_fit_error", 0.0) if isinstance(d, dict) else 0.0,
                    metadata=r.metadata,
                ))
            except Exception as exc:
                logger.warning("engine_failed", extra={
                    "engine": eng.name, "error": str(exc)})
        return out

    def _resolve_weights(
        self, regime: str, perceptions: List[EnginePerception],
    ) -> Dict[str, float]:
        names = [p.engine_name for p in perceptions]
        if self._plasticity and self._plasticity.has_history(regime):
            return self._plasticity.get_weights(regime, names)
        return {n: self._base_weights.get(n, 1.0 / len(names)) for n in names}

    def _fallback(
        self,
        values: List[float],
        profile: StructuralAnalysis,
        builder: ExplanationBuilder,
        timer: Optional[PipelineTimer] = None,
        reason: str = "no_valid_perceptions",
    ) -> PredictionResult:
        tail = values[-min(3, len(values)):] if values else [0.0]
        predicted = sum(tail) / len(tail)
        builder.set_fallback(predicted, reason=reason)

        fallback_reason = reason
        diag = MetaDiagnostic(
            signal_profile=profile, perceptions=[], inhibition_states=[],
            final_weights={}, selected_engine="none",
            selection_reason="all_engines_failed",
            fusion_method="fallback", fallback_reason=fallback_reason,
        )
        self._last_diagnostic = diag
        self._last_explanation = builder.build()
        self._last_regime = profile.regime.value
        self._last_perceptions = []

        metadata: dict = {
            "cognitive_diagnostic": diag.to_dict(),
            "explanation": self._last_explanation.to_dict(),
        }
        if timer is not None:
            metadata["pipeline_timing"] = timer.to_dict()

        return PredictionResult(
            predicted_value=predicted, confidence=0.2, trend="stable",
            metadata=metadata,
        )
