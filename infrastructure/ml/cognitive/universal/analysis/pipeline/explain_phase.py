"""Explain phase: assemble Explanation domain object."""

from __future__ import annotations

import time
from typing import Any, Dict

from iot_machine_learning.infrastructure.ml.cognitive.explanation import ExplanationBuilder


class ExplainPhase:
    """Phase 5: Assemble Explanation domain object."""

    def execute(
        self,
        builder: ExplanationBuilder,
        fused_value: float,
        fused_confidence: float,
        fused_trend: str,
        final_weights: Dict[str, float],
        selected: str,
        reason: str,
        method: str,
        timing: Dict[str, float],
    ) -> Any:
        """Execute explain phase.

        Args:
            builder: ExplanationBuilder instance
            fused_value: Fused prediction value
            fused_confidence: Fused confidence
            fused_trend: Fused trend
            final_weights: Final engine weights
            selected: Selected engine
            reason: Selection reason
            method: Fusion method
            timing: Pipeline timing dict

        Returns:
            Explanation domain object
        """
        t0 = time.monotonic()
        
        builder.set_fusion(
            fused_value, fused_confidence, fused_trend,
            final_weights, selected, reason, method
        )
        
        explanation = builder.build()
        
        timing["explain"] = (time.monotonic() - t0) * 1000
        
        return explanation
