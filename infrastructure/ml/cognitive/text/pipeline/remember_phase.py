"""Text remember phase: recall similar past documents from cognitive memory."""

from __future__ import annotations

import time
from typing import Any, Dict

from iot_machine_learning.infrastructure.ml.cognitive.text.memory_enricher import TextMemoryEnricher


class TextRememberPhase:
    """Phase 3: Recall similar past documents from cognitive memory."""

    def __init__(self) -> None:
        self._enricher = TextMemoryEnricher()

    def execute(
        self,
        full_text: str,
        domain: str,
        cognitive_memory: Any,
        document_id: str,
        timing: Dict[str, float],
    ) -> Any:
        """Execute remember phase.

        Args:
            full_text: Complete document text
            domain: Classified domain
            cognitive_memory: Cognitive memory port
            document_id: Document identifier
            timing: Pipeline timing dict

        Returns:
            Recall context object
        """
        t0 = time.monotonic()
        
        recall_ctx = self._enricher.enrich(
            full_text=full_text,
            domain=domain,
            cognitive_memory=cognitive_memory,
            document_id=document_id,
        )
        
        recall_ms = (time.monotonic() - t0) * 1000
        timing["remember"] = recall_ms
        
        return recall_ctx
