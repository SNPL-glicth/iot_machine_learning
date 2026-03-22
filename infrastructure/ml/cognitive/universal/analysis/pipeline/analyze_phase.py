"""Analyze phase: collect perceptions from sub-analyzers."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from iot_machine_learning.infrastructure.ml.cognitive.universal.analysis.perception_collector import UniversalPerceptionCollector
from iot_machine_learning.infrastructure.ml.cognitive.universal.analysis.types import InputType, UniversalContext


class AnalyzePhase:
    """Phase 2: Collect perceptions from type-specific sub-analyzers."""

    def __init__(self) -> None:
        self._collector = UniversalPerceptionCollector()

    def execute(
        self,
        raw_data: Any,
        input_type: InputType,
        metadata: Dict[str, Any],
        pre_computed_scores: Optional[Dict[str, Any]],
        timing: Dict[str, float],
    ) -> List:
        """Execute analyze phase.

        Args:
            raw_data: Any input
            input_type: Detected input type
            metadata: Input metadata
            pre_computed_scores: Optional pre-computed analysis scores
            timing: Pipeline timing dict

        Returns:
            List of EnginePerception objects
        """
        t0 = time.monotonic()
        
        perceptions = self._collector.collect(
            raw_data, input_type, metadata, pre_computed_scores
        )
        
        # Apply pattern plasticity weights if enabled
        # TODO: Implement _apply_pattern_weights method
        # if self._plasticity:
        #     perceptions = self._apply_pattern_weights(
        #         perceptions, metadata["domain"], input_type
        #     )
        
        timing["analyze"] = (time.monotonic() - t0) * 1000
        
        return perceptions
