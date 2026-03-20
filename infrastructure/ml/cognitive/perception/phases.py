"""Perception phase setters for ExplanationBuilder.

Covers the early pipeline phases:
- PERCEIVE (signal analysis)
- FILTER  (optional filtering)
- PREDICT (engine perceptions)
- ADAPT   (plasticity adaptation)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from iot_machine_learning.domain.entities.explainability.contribution_breakdown import (
    EngineContribution,
)
from iot_machine_learning.domain.entities.explainability.reasoning_trace import (
    PhaseKind,
    ReasoningPhase,
)
from iot_machine_learning.domain.entities.explainability.signal_snapshot import (
    FilterSnapshot,
    SignalSnapshot,
)
from iot_machine_learning.domain.entities.series.structural_analysis import (
    StructuralAnalysis,
)

from ..analysis.types import EnginePerception

if TYPE_CHECKING:
    from ..explanation.builder import ExplanationBuilder


def set_signal(
    builder: ExplanationBuilder,
    profile: StructuralAnalysis,
) -> ExplanationBuilder:
    """Registra la fase PERCEIVE."""
    regime_str = profile.regime.value if hasattr(profile.regime, 'value') else str(profile.regime)
    builder._signal = SignalSnapshot(
        n_points=profile.n_points,
        mean=profile.mean,
        std=profile.std,
        noise_ratio=profile.noise_ratio,
        slope=profile.slope,
        curvature=profile.curvature,
        regime=regime_str,
        dt=profile.dt,
    )
    builder._regime = regime_str
    builder._phases.append(ReasoningPhase(
        kind=PhaseKind.PERCEIVE,
        summary={
            "n_points": profile.n_points,
            "regime": regime_str,
            "noise_ratio": round(profile.noise_ratio, 4),
        },
        outputs={"signal_profile": profile.to_dict()},
    ))
    return builder


def set_filter(
    builder: ExplanationBuilder,
    filter_name: str,
    diagnostic: Optional[dict] = None,
) -> ExplanationBuilder:
    """Registra la fase FILTER (solo si se aplicó filtrado)."""
    diag = diagnostic or {}
    builder._filter = FilterSnapshot(
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
    builder._phases.append(ReasoningPhase(
        kind=PhaseKind.FILTER,
        summary={
            "filter_name": filter_name,
            "noise_reduction": round(
                diag.get("noise_reduction_ratio", 0.0), 4
            ),
        },
        outputs={"filter_diagnostic": diag},
    ))
    return builder


def set_perceptions(
    builder: ExplanationBuilder,
    perceptions: List[EnginePerception],
    n_engines_total: int = 0,
) -> ExplanationBuilder:
    """Registra la fase PREDICT (solo si hubo engines que respondieron)."""
    builder._n_engines_available = n_engines_total or len(perceptions)
    if not perceptions:
        return builder

    builder._phases.append(ReasoningPhase(
        kind=PhaseKind.PREDICT,
        summary={
            "n_engines_responded": len(perceptions),
            "n_engines_available": builder._n_engines_available,
            "engines": [p.engine_name for p in perceptions],
        },
        outputs={
            "predictions": {
                p.engine_name: round(p.predicted_value, 6)
                for p in perceptions
            },
        },
    ))
    # Pre-populate contributions (weights filled later by fusion phase)
    builder._contributions = [
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
    return builder


def set_adaptation(
    builder: ExplanationBuilder,
    adapted: bool,
    regime: str,
    weights_source: str = "plasticity",
) -> ExplanationBuilder:
    """Registra la fase ADAPT (solo si la plasticidad participó)."""
    if not adapted:
        return builder

    builder._phases.append(ReasoningPhase(
        kind=PhaseKind.ADAPT,
        summary={
            "adapted": True,
            "regime": regime,
            "weights_source": weights_source,
        },
    ))
    return builder
