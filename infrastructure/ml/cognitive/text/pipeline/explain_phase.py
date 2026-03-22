"""Text explain phase: assemble Explanation domain object."""

from __future__ import annotations

import time
from typing import Any, Dict, List

from iot_machine_learning.infrastructure.ml.cognitive.text.explanation_assembler import TextExplanationAssembler


class TextExplainPhase:
    """Phase 5: Assemble Explanation domain object."""

    def __init__(self) -> None:
        self._assembler = TextExplanationAssembler()

    def execute(
        self,
        document_id: str,
        signal: Any,
        perceptions: List[Any],
        inhibition_states: List[Any],
        final_weights: Dict[str, float],
        selected_engine: str,
        selection_reason: str,
        fusion_method: str,
        fused_confidence: float,
        domain: str,
        severity: Any,
        pipeline_phases: List[Dict[str, Any]],
        timing: Dict[str, float],
    ) -> Any:
        """Execute explain phase.

        Args:
            document_id: Document identifier
            signal: Signal profile
            perceptions: Engine perceptions
            inhibition_states: Inhibition states
            final_weights: Final engine weights
            selected_engine: Selected engine
            selection_reason: Selection reason
            fusion_method: Fusion method used
            fused_confidence: Fused confidence
            domain: Document domain
            severity: Severity result
            pipeline_phases: Previous phase summaries
            timing: Pipeline timing dict

        Returns:
            Explanation domain object
        """
        t0 = time.monotonic()
        
        explanation = self._assembler.assemble(
            document_id=document_id,
            signal=signal,
            perceptions=perceptions,
            inhibition_states=inhibition_states,
            final_weights=final_weights,
            selected_engine=selected_engine,
            selection_reason=selection_reason,
            fusion_method=fusion_method,
            fused_confidence=fused_confidence,
            domain=domain,
            severity=severity,
            pipeline_phases=pipeline_phases,
        )
        
        explain_ms = (time.monotonic() - t0) * 1000
        timing["explain"] = explain_ms
        
        return explanation
