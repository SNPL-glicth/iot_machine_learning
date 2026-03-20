"""Orchestrator Helper Functions.

Auxiliary functions extracted from MetaCognitiveOrchestrator to reduce complexity.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from ..analysis.types import EnginePerception, PipelineTimer
from ..explanation.explanation_builder import ExplanationBuilder
from ..interfaces import PredictionResult
from ....domain.entities.series.structural_analysis import StructuralAnalysis

logger = logging.getLogger(__name__)


def collect_perceptions(
    engines: List,
    values: List[float],
    timestamps: Optional[List[float]],
) -> List[EnginePerception]:
    """Collect predictions from all capable engines.
    
    Args:
        engines: List of PredictionEngine instances
        values: Time series values
        timestamps: Optional timestamps
    
    Returns:
        List of EnginePerception objects
    """
    out: List[EnginePerception] = []
    for eng in engines:
        if not eng.can_handle(len(values)):
            continue
        try:
            r = eng.predict(values, timestamps)
            d = r.metadata.get("diagnostic", {}) or {}
            out.append(EnginePerception(
                engine_name=eng.name,
                predicted_value=r.predicted_value,
                confidence=r.confidence,
                trend=r.trend,
                stability=d.get("stability_indicator", 0.0) if isinstance(d, dict) else 0.0,
                local_fit_error=d.get("local_fit_error", 0.0) if isinstance(d, dict) else 0.0,
                metadata=r.metadata,
            ))
        except Exception as exc:
            logger.warning("engine_failed", extra={
                "engine": eng.name, "error": str(exc)})
    return out


def create_fallback_result(
    values: List[float],
    profile: StructuralAnalysis,
    builder: ExplanationBuilder,
    timer: Optional[PipelineTimer] = None,
    reason: str = "no_valid_perceptions",
) -> PredictionResult:
    """Create fallback prediction result when all engines fail.
    
    Args:
        values: Time series values
        profile: Signal profile
        builder: ExplanationBuilder instance
        timer: Optional pipeline timer
        reason: Reason for fallback
    
    Returns:
        PredictionResult with fallback prediction
    """
    from .analysis.types import MetaDiagnostic
    
    tail = values[-min(3, len(values)):] if values else [0.0]
    predicted = sum(tail) / len(tail)
    builder.set_fallback(predicted, reason=reason)

    fallback_reason = reason
    diag = MetaDiagnostic(
        signal_profile=profile,
        perceptions=[],
        inhibition_states=[],
        final_weights={},
        selected_engine="none",
        selection_reason="all_engines_failed",
        fusion_method="fallback",
        fallback_reason=fallback_reason,
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
