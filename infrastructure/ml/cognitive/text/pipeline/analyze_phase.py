"""Text analyze phase: map sub-analyzer scores to EnginePerception[]."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

from iot_machine_learning.infrastructure.ml.cognitive.text.perception_collector import TextPerceptionCollector
from iot_machine_learning.infrastructure.ml.cognitive.text.types import TextAnalysisInput


class TextAnalyzePhase:
    """Phase 2: Map sub-analyzer scores to EnginePerception[]."""

    def __init__(self) -> None:
        self._collector = TextPerceptionCollector()

    def execute(
        self,
        inp: TextAnalysisInput,
        timing: Dict[str, float],
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Execute analyze phase.

        Args:
            inp: Pre-computed text analysis scores
            timing: Pipeline timing dict

        Returns:
            Tuple of (perceptions, phase_summary)
        """
        t0 = time.monotonic()
        
        perceptions = self._collector.collect(inp)
        
        predict_ms = (time.monotonic() - t0) * 1000
        timing["predict"] = predict_ms
        
        phase_summary = {
            "kind": "predict",
            "summary": {"n_engines": len(perceptions)},
            "duration_ms": predict_ms,
        }

        return perceptions, phase_summary
