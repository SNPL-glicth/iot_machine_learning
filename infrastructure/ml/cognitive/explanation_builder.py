"""ExplanationBuilder — assembles domain Explanation from orchestrator outputs.

Responsabilidad ÚNICA: traducir tipos de infraestructura cognitiva
(StructuralAnalysis, EnginePerception, InhibitionState, MetaDiagnostic)
a tipos de dominio de explicabilidad (Explanation, ReasoningTrace, etc.).

El orquestador NO construye la explicación directamente.
Solo pasa sus outputs al builder.

Las fases se agregan dinámicamente según lo que realmente ocurrió:
- PERCEIVE siempre se agrega (siempre hay análisis de señal).
- FILTER solo si se proporcionó un FilterDiagnostic.
- PREDICT solo si hubo engines que respondieron.
- ADAPT solo si la plasticidad participó.
- INHIBIT solo si algún engine fue suprimido.
- FUSE solo si hubo fusión (>0 engines activos).

Esto NO es un pipeline fijo. Es un registro de lo que emergió.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from iot_machine_learning.domain.entities.explainability.contribution_breakdown import (
    ContributionBreakdown,
    EngineContribution,
)
from iot_machine_learning.domain.entities.explainability.explanation import (
    Explanation,
    Outcome,
)
from iot_machine_learning.domain.entities.explainability.reasoning_trace import (
    PhaseKind,
    ReasoningPhase,
    ReasoningTrace,
)
from iot_machine_learning.domain.entities.explainability.signal_snapshot import (
    FilterSnapshot,
    SignalSnapshot,
)

from iot_machine_learning.domain.entities.series.structural_analysis import (
    StructuralAnalysis,
)

from .types import EnginePerception, InhibitionState, MetaDiagnostic


class ExplanationBuilder:
    """Construye un ``Explanation`` a partir de los outputs del orquestador.

    Uso típico::

        builder = ExplanationBuilder(series_id="temp_room_1")
        builder.set_signal(profile)
        builder.set_filter(filter_diagnostic_dict)  # opcional
        builder.set_perceptions(perceptions)
        builder.set_inhibition(inh_states, base_weights)
        builder.set_adaptation(adapted=True, regime="stable")  # opcional
        builder.set_fusion(fused_val, fused_conf, fused_trend,
                           final_weights, selected, reason, method)
        explanation = builder.build()
    """

    def __init__(self, series_id: str) -> None:
        self._series_id = series_id
        self._signal: Optional[SignalSnapshot] = None
        self._filter: Optional[FilterSnapshot] = None
        self._phases: List[ReasoningPhase] = []
        self._contributions: List[EngineContribution] = []
        self._fusion_method: str = "weighted_average"
        self._selected_engine: str = "none"
        self._selection_reason: str = ""
        self._fallback_used: bool = False
        self._fallback_reason: Optional[str] = None
        self._outcome: Optional[Outcome] = None
        self._audit_trace_id: Optional[str] = None
        self._n_engines_available: int = 0
        self._n_engines_active: int = 0
        self._regime: str = "unknown"

    # ── Phase setters (each adds a phase only if relevant) ──────

    def set_signal(self, profile: StructuralAnalysis) -> ExplanationBuilder:
        """Registra la fase PERCEIVE."""
        regime_str = profile.regime.value
        self._signal = SignalSnapshot(
            n_points=profile.n_points,
            mean=profile.mean,
            std=profile.std,
            noise_ratio=profile.noise_ratio,
            slope=profile.slope,
            curvature=profile.curvature,
            regime=regime_str,
            dt=profile.dt,
        )
        self._regime = regime_str
        self._phases.append(ReasoningPhase(
            kind=PhaseKind.PERCEIVE,
            summary={
                "n_points": profile.n_points,
                "regime": regime_str,
                "noise_ratio": round(profile.noise_ratio, 4),
            },
            outputs={"signal_profile": profile.to_dict()},
        ))
        return self

    def set_filter(
        self,
        filter_name: str,
        diagnostic: Optional[dict] = None,
    ) -> ExplanationBuilder:
        """Registra la fase FILTER (solo si se aplicó filtrado)."""
        diag = diagnostic or {}
        self._filter = FilterSnapshot(
            filter_name=filter_name,
            n_points=diag.get("n_points", 0),
            noise_reduction_ratio=diag.get("noise_reduction_ratio", 0.0),
            mean_absolute_error=diag.get("mean_absolute_error", 0.0),
            max_absolute_error=diag.get("max_absolute_error", 0.0),
            lag_estimate=diag.get("lag_estimate", 0),
            signal_distortion=diag.get("signal_distortion", 0.0),
            is_effective=diag.get("noise_reduction_ratio", 0.0) > 0.05
            and diag.get("signal_distortion", 0.0) < 0.5,
        )
        self._phases.append(ReasoningPhase(
            kind=PhaseKind.FILTER,
            summary={
                "filter_name": filter_name,
                "noise_reduction": round(
                    diag.get("noise_reduction_ratio", 0.0), 4
                ),
            },
            outputs={"filter_diagnostic": diag},
        ))
        return self

    def set_perceptions(
        self,
        perceptions: List[EnginePerception],
        n_engines_total: int = 0,
    ) -> ExplanationBuilder:
        """Registra la fase PREDICT (solo si hubo engines que respondieron)."""
        self._n_engines_available = n_engines_total or len(perceptions)
        if not perceptions:
            return self

        self._phases.append(ReasoningPhase(
            kind=PhaseKind.PREDICT,
            summary={
                "n_engines_responded": len(perceptions),
                "n_engines_available": self._n_engines_available,
                "engines": [p.engine_name for p in perceptions],
            },
            outputs={
                "predictions": {
                    p.engine_name: round(p.predicted_value, 6)
                    for p in perceptions
                },
            },
        ))
        # Pre-populate contributions (weights filled later)
        self._contributions = [
            EngineContribution(
                engine_name=p.engine_name,
                predicted_value=p.predicted_value,
                confidence=p.confidence,
                trend=p.trend,
                stability=p.stability,
                local_fit_error=p.local_fit_error,
            )
            for p in perceptions
        ]
        return self

    def set_adaptation(
        self,
        adapted: bool,
        regime: str,
        weights_source: str = "plasticity",
    ) -> ExplanationBuilder:
        """Registra la fase ADAPT (solo si la plasticidad participó)."""
        if not adapted:
            return self

        self._phases.append(ReasoningPhase(
            kind=PhaseKind.ADAPT,
            summary={
                "adapted": True,
                "regime": regime,
                "weights_source": weights_source,
            },
        ))
        return self

    def set_inhibition(
        self,
        inh_states: List[InhibitionState],
        base_weights: Dict[str, float],
    ) -> ExplanationBuilder:
        """Registra la fase INHIBIT (solo si algún engine fue suprimido)."""
        any_inhibited = any(s.suppression_factor > 0.01 for s in inh_states)

        if any_inhibited:
            self._phases.append(ReasoningPhase(
                kind=PhaseKind.INHIBIT,
                summary={
                    "n_inhibited": sum(
                        1 for s in inh_states if s.suppression_factor > 0.01
                    ),
                    "reasons": {
                        s.engine_name: s.inhibition_reason
                        for s in inh_states
                        if s.suppression_factor > 0.01
                    },
                },
                inputs={"base_weights": base_weights},
                outputs={
                    "inhibited_weights": {
                        s.engine_name: round(s.inhibited_weight, 4)
                        for s in inh_states
                    },
                },
            ))

        # Update contributions with inhibition data
        inh_map = {s.engine_name: s for s in inh_states}
        updated: List[EngineContribution] = []
        for c in self._contributions:
            inh = inh_map.get(c.engine_name)
            if inh:
                updated.append(EngineContribution(
                    engine_name=c.engine_name,
                    predicted_value=c.predicted_value,
                    confidence=c.confidence,
                    trend=c.trend,
                    base_weight=inh.base_weight,
                    final_weight=inh.inhibited_weight,
                    inhibited=inh.suppression_factor > 0.01,
                    inhibition_reason=inh.inhibition_reason,
                    local_fit_error=c.local_fit_error,
                    stability=c.stability,
                ))
            else:
                bw = base_weights.get(c.engine_name, 0.0)
                updated.append(EngineContribution(
                    engine_name=c.engine_name,
                    predicted_value=c.predicted_value,
                    confidence=c.confidence,
                    trend=c.trend,
                    base_weight=bw,
                    final_weight=bw,
                    local_fit_error=c.local_fit_error,
                    stability=c.stability,
                ))
        self._contributions = updated
        return self

    def set_fusion(
        self,
        fused_value: float,
        fused_confidence: float,
        fused_trend: str,
        final_weights: Dict[str, float],
        selected_engine: str,
        selection_reason: str,
        fusion_method: str = "weighted_average",
    ) -> ExplanationBuilder:
        """Registra la fase FUSE."""
        self._fusion_method = fusion_method
        self._selected_engine = selected_engine
        self._selection_reason = selection_reason
        self._n_engines_active = sum(
            1 for w in final_weights.values() if w > 0.01
        )

        self._phases.append(ReasoningPhase(
            kind=PhaseKind.FUSE,
            summary={
                "method": fusion_method,
                "selected_engine": selected_engine,
                "n_active": self._n_engines_active,
            },
            inputs={"final_weights": {
                k: round(v, 4) for k, v in final_weights.items()
            }},
            outputs={
                "fused_value": round(fused_value, 6),
                "fused_confidence": round(fused_confidence, 4),
                "fused_trend": fused_trend,
            },
        ))

        # Update contributions with final normalized weights
        updated: List[EngineContribution] = []
        for c in self._contributions:
            fw = final_weights.get(c.engine_name, c.final_weight)
            updated.append(EngineContribution(
                engine_name=c.engine_name,
                predicted_value=c.predicted_value,
                confidence=c.confidence,
                trend=c.trend,
                base_weight=c.base_weight,
                final_weight=fw,
                inhibited=c.inhibited,
                inhibition_reason=c.inhibition_reason,
                local_fit_error=c.local_fit_error,
                stability=c.stability,
            ))
        self._contributions = updated

        self._outcome = Outcome(
            kind="prediction",
            predicted_value=fused_value,
            confidence=fused_confidence,
            trend=fused_trend,
        )
        return self

    def set_fallback(
        self,
        predicted_value: float,
        reason: str,
    ) -> ExplanationBuilder:
        """Registra un fallback (sin engines activos)."""
        self._fallback_used = True
        self._fallback_reason = reason
        self._selected_engine = "none"
        self._fusion_method = "fallback"
        self._outcome = Outcome(
            kind="prediction",
            predicted_value=predicted_value,
            confidence=0.2,
            trend="stable",
            extra={"fallback_reason": reason},
        )
        return self

    def set_audit_trace_id(self, trace_id: str) -> ExplanationBuilder:
        self._audit_trace_id = trace_id
        return self

    # ── Build ───────────────────────────────────────────────────

    def build(self) -> Explanation:
        """Construye el ``Explanation`` final.

        Las fases incluidas reflejan exactamente lo que ocurrió.
        No hay fases ficticias.
        """
        return Explanation(
            series_id=self._series_id,
            signal=self._signal or SignalSnapshot.empty(),
            filter=self._filter or FilterSnapshot.empty(),
            contributions=ContributionBreakdown(
                contributions=self._contributions,
                fusion_method=self._fusion_method,
                selected_engine=self._selected_engine,
                selection_reason=self._selection_reason,
                fallback_used=self._fallback_used,
                fallback_reason=self._fallback_reason,
            ),
            trace=ReasoningTrace(
                phases=self._phases,
                regime_at_inference=self._regime,
                n_engines_available=self._n_engines_available,
                n_engines_active=self._n_engines_active,
            ),
            outcome=self._outcome or Outcome(),
            audit_trace_id=self._audit_trace_id,
        )
