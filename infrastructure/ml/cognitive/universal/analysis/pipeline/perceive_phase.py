"""Perceive phase: detect type, classify domain, profile signal."""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple

from iot_machine_learning.infrastructure.ml.cognitive.explanation import ExplanationBuilder
from iot_machine_learning.infrastructure.ml.cognitive.universal.analysis.input_detector import detect_input_type
from iot_machine_learning.infrastructure.ml.cognitive.universal.analysis.domain_classifier import classify_domain
from iot_machine_learning.infrastructure.ml.cognitive.universal.analysis.signal_profiler import UniversalSignalProfiler
from iot_machine_learning.infrastructure.ml.cognitive.universal.analysis.types import InputType, UniversalContext


class PerceivePhase:
    """Phase 1: Detect type, classify domain, build signal profile."""

    def __init__(self) -> None:
        self._profiler = UniversalSignalProfiler()

    def execute(
        self,
        raw_data: Any,
        ctx: UniversalContext,
        timing: Dict[str, float],
    ) -> Tuple[InputType, Dict[str, Any], str, Any, ExplanationBuilder]:
        """Execute perceive phase.

        Args:
            raw_data: Any input (str, List[float], Dict, etc.)
            ctx: Pipeline configuration and environment
            timing: Pipeline timing dict

        Returns:
            Tuple of (input_type, metadata, domain, signal, builder)
        """
        t0 = time.monotonic()
        
        input_type, metadata = detect_input_type(raw_data)
        
        domain = classify_domain(
            raw_data, input_type, metadata, ctx.domain_hint
        )
        
        signal = self._profiler.profile(
            raw_data, input_type, metadata, domain
        )
        
        builder = ExplanationBuilder(ctx.series_id)
        builder.set_signal(signal)
        
        timing["perceive"] = (time.monotonic() - t0) * 1000
        
        return input_type, metadata, domain, signal, builder
