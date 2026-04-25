"""Fallback prediction result when all engines fail."""

from __future__ import annotations

from typing import Optional

from ..analysis.types import EnginePerception, PipelineTimer
from ..explanation.explanation_builder import ExplanationBuilder
from ...interfaces import PredictionResult
from iot_machine_learning.domain.entities.series.structural_analysis import StructuralAnalysis


def create_fallback_result(
    values: list[float],
    profile: StructuralAnalysis,
    builder: ExplanationBuilder,
    timer: Optional[PipelineTimer] = None,
    reason: str = "no_valid_perceptions",
) -> PredictionResult:
    """Create fallback prediction result when all engines fail."""
    from ..analysis.types import MetaDiagnostic

    tail = values[-min(3, len(values)):] if values else [0.0]
    predicted = sum(tail) / len(tail)
    builder.set_fallback(predicted, reason=reason)

    diag = MetaDiagnostic(
        signal_profile=profile,
        perceptions=[],
        inhibition_states=[],
        final_weights={},
        selected_engine="none",
        selection_reason="all_engines_failed",
        fusion_method="fallback",
        fallback_reason=reason,
    )

    explanation = builder.build()

    metadata: dict = {
        "cognitive_diagnostic": diag.to_dict(),
        "explanation": explanation.to_dict(),
    }

    if timer:
        metadata["pipeline_timing"] = timer.to_dict()

    return PredictionResult(
        predicted_value=predicted,
        confidence=0.2,
        trend="unknown",
        metadata=metadata,
    )
