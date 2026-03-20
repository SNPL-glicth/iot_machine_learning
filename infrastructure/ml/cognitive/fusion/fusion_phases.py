"""Fusion phase setters for ExplanationBuilder.

Covers the late pipeline phases:
- INHIBIT (weight suppression)
- FUSE    (weighted fusion)
- fallback (no active engines)
- audit_trace_id (metadata)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from iot_machine_learning.domain.entities.explainability.contribution_breakdown import (
    EngineContribution,
)
from iot_machine_learning.domain.entities.explainability.explanation import Outcome
from iot_machine_learning.domain.entities.explainability.reasoning_trace import (
    PhaseKind,
    ReasoningPhase,
)

from ..analysis.types import InhibitionState

if TYPE_CHECKING:
    from ..explanation.builder import ExplanationBuilder


def set_inhibition(
    builder: ExplanationBuilder,
    inh_states: List[InhibitionState],
    base_weights: Dict[str, float],
) -> ExplanationBuilder:
    """Registra la fase INHIBIT (solo si algún engine fue suprimido)."""
    any_inhibited = any(s.suppression_factor > 0.01 for s in inh_states)

    if any_inhibited:
        builder._phases.append(ReasoningPhase(
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
    for c in builder._contributions:
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
    builder._contributions = updated
    return builder


def set_fusion(
    builder: ExplanationBuilder,
    fused_value: float,
    fused_confidence: float,
    fused_trend: str,
    final_weights: Dict[str, float],
    selected_engine: str,
    selection_reason: str,
    fusion_method: str = "weighted_average",
) -> ExplanationBuilder:
    """Registra la fase FUSE."""
    builder._fusion_method = fusion_method
    builder._selected_engine = selected_engine
    builder._selection_reason = selection_reason
    builder._n_engines_active = sum(
        1 for w in final_weights.values() if w > 0.01
    )

    builder._phases.append(ReasoningPhase(
        kind=PhaseKind.FUSE,
        summary={
            "method": fusion_method,
            "selected_engine": selected_engine,
            "n_active": builder._n_engines_active,
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
    for c in builder._contributions:
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
    builder._contributions = updated

    builder._outcome = Outcome(
        kind="prediction",
        predicted_value=fused_value,
        confidence=fused_confidence,
        trend=fused_trend,
    )
    return builder


def set_fallback(
    builder: ExplanationBuilder,
    predicted_value: float,
    reason: str,
) -> ExplanationBuilder:
    """Registra un fallback (sin engines activos)."""
    builder._fallback_used = True
    builder._fallback_reason = reason
    builder._selected_engine = "none"
    builder._fusion_method = "fallback"
    builder._outcome = Outcome(
        kind="prediction",
        predicted_value=predicted_value,
        confidence=0.2,
        trend="stable",
        extra={"fallback_reason": reason},
    )
    return builder


def set_audit_trace_id(
    builder: ExplanationBuilder,
    trace_id: str,
) -> ExplanationBuilder:
    """Registra el audit trace ID."""
    builder._audit_trace_id = trace_id
    return builder
