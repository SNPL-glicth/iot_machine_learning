from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List
    from ...interfaces import PredictionResult
    from ..explanation import ExplanationBuilder
    from ..analysis.types import MetaDiagnostic, PipelineTimer
    from iot_machine_learning.domain.entities.series.structural_analysis import StructuralAnalysis

from ..perception.helpers import create_fallback_result
from ..analysis.types import MetaDiagnostic


def handle_fallback(
    values: List[float],
    profile: StructuralAnalysis,
    builder: ExplanationBuilder,
    timer: PipelineTimer,
    reason: str,
) -> tuple[PredictionResult, MetaDiagnostic, object, str, list]:
    """Handle fallback case and create fallback result.
    
    Returns:
        Tuple of (result, diagnostic, explanation, regime_str, empty_perceptions)
    """
    result = create_fallback_result(values, profile, builder, timer, reason)
    diagnostic = MetaDiagnostic(
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
    regime_str = profile.regime.value
    
    return result, diagnostic, explanation, regime_str, []
