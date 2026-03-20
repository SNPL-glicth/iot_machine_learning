"""ExplanationBuilder — core class that assembles domain Explanation.

Delegates all phase-setter logic to phase_setters.py.
This file contains only: constructor, internal state, and build().
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

from .analysis.types import EnginePerception, InhibitionState
from . import phase_setters


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

    # ── Phase setters (delegate to phase_setters module) ────────

    def set_signal(self, profile: StructuralAnalysis) -> ExplanationBuilder:
        """Registra la fase PERCEIVE."""
        return phase_setters.set_signal(self, profile)

    def set_filter(
        self,
        filter_name: str,
        diagnostic: Optional[dict] = None,
    ) -> ExplanationBuilder:
        """Registra la fase FILTER (solo si se aplicó filtrado)."""
        return phase_setters.set_filter(self, filter_name, diagnostic)

    def set_perceptions(
        self,
        perceptions: List[EnginePerception],
        n_engines_total: int = 0,
    ) -> ExplanationBuilder:
        """Registra la fase PREDICT (solo si hubo engines que respondieron)."""
        return phase_setters.set_perceptions(self, perceptions, n_engines_total)

    def set_adaptation(
        self,
        adapted: bool,
        regime: str,
        weights_source: str = "plasticity",
    ) -> ExplanationBuilder:
        """Registra la fase ADAPT (solo si la plasticidad participó)."""
        return phase_setters.set_adaptation(self, adapted, regime, weights_source)

    def set_inhibition(
        self,
        inh_states: List[InhibitionState],
        base_weights: Dict[str, float],
    ) -> ExplanationBuilder:
        """Registra la fase INHIBIT (solo si algún engine fue suprimido)."""
        return phase_setters.set_inhibition(self, inh_states, base_weights)

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
        return phase_setters.set_fusion(
            self,
            fused_value,
            fused_confidence,
            fused_trend,
            final_weights,
            selected_engine,
            selection_reason,
            fusion_method,
        )

    def set_fallback(
        self,
        predicted_value: float,
        reason: str,
    ) -> ExplanationBuilder:
        """Registra un fallback (sin engines activos)."""
        return phase_setters.set_fallback(self, predicted_value, reason)

    def set_audit_trace_id(self, trace_id: str) -> ExplanationBuilder:
        return phase_setters.set_audit_trace_id(self, trace_id)

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
