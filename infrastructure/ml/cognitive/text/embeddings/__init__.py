"""HybridEntityDetector — entity extraction with optional vector memory enrichment.

Design:
  ML_ENABLE_HYBRID_EMBEDDINGS=false (default):
    Pure regex passthrough — delegates to RegexEntityExtractor without
    duplicating its logic. Zero new dependencies.

  ML_ENABLE_HYBRID_EMBEDDINGS=true:
    Uses VectorMemoryPort to retrieve semantically similar stored documents,
    then extracts entities from those documents. The result is the union of
    input-text entities and semantically-backed stored-document entities, deduplicated.

  On any error (unavailable, timeout, no data): safe fallback
  to regex-only result. Never raises, never returns None.

  Pure algorithmic layer — no network calls, no HTTP requests, no database drivers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from iot_machine_learning.domain.ports.vector_memory_port import VectorMemoryPort
from iot_machine_learning.infrastructure.ml.cognitive.text.semantic_extraction.composite_entity_extractor import (
    RegexEntityExtractor,
)

logger = logging.getLogger(__name__)


@dataclass
class EntityResult:
    """Result container matching the HybridEntityDetector contract.

    Attributes:
        entities: List of SemanticEntity objects.
    """

    entities: list[Any] = field(default_factory=list)

    def to_list(self) -> list[str]:
        """Return entity texts as a flat string list."""
        return [str(e.text) if hasattr(e, "text") else str(e) for e in self.entities]

    @property
    def count(self) -> int:
        return len(self.entities)


class HybridEntityDetector:
    """Entity detector with hybrid regex + VectorMemoryPort enrichment.

    Args:
        domain_hint: Domain context passed to RegexEntityExtractor.
        magnitude_threshold: Minimum semantic certainty for vector-backed
            entities (converted to min_certainty = 1 - threshold).
        vector_memory: Optional VectorMemoryPort for semantic retrieval.
        hybrid_enabled: Explicit boolean to enable hybrid vector memory search.
    """

    def __init__(
        self,
        domain_hint: str = "general",
        magnitude_threshold: float = 0.3,
        vector_memory: Optional[VectorMemoryPort] = None,
        hybrid_enabled: bool = False,
    ) -> None:
        self._domain = domain_hint
        self._threshold = magnitude_threshold
        self._vector_memory = vector_memory
        self._hybrid_enabled = hybrid_enabled
        self._regex = RegexEntityExtractor(domain_hint=domain_hint)

    # ── Public API ──────────────────────────────────────────────

    def extract_entities(self, text: str) -> EntityResult:
        """Extract entities using regex or hybrid regex+vector memory.

        Steps:
          1. Always extract regex entities from the input text.
          2. If vector memory is not enabled or not provided, return regex result.
          3. If enabled, query vector memory for semantically similar
             stored documents, extract entities from their text,
             merge with input entities, deduplicate.
          4. On any error, safe fallback to regex-only result.

        Args:
            text: Input text to analyze.

        Returns:
            EntityResult with .to_list() -> list[str].
        """
        if not text or not text.strip():
            return EntityResult()

        regex_entities = self._extract_regex(text)

        if not self._hybrid_enabled or self._vector_memory is None:
            return EntityResult(regex_entities)

        try:
            vector_entities = self._enrich_via_vector_memory(text)
            if not vector_entities:
                return EntityResult(regex_entities)
            merged = self._merge_entity_lists(regex_entities, vector_entities)
            return EntityResult(merged)
        except Exception as exc:
            logger.debug("hybrid_vector_fallback: %s", exc)
            return EntityResult(regex_entities)

    # ── Internals ───────────────────────────────────────────────

    def _extract_regex(self, text: str) -> list[Any]:
        result = self._regex.extract(text)
        return list(result.entities)

    def _enrich_via_vector_memory(self, text: str) -> list[Any]:
        if self._vector_memory is None:
            return []

        concept = " ".join(text.split()[:200])
        min_certainty = max(0.0, 1.0 - self._threshold)

        explanation_texts = self._vector_memory.search_similar_explanations(
            concept=concept,
            min_certainty=min_certainty,
            limit=5,
        )

        entities: list[Any] = []
        for expl_text in explanation_texts:
            if expl_text:
                result = self._regex.extract(expl_text)
                entities.extend(result.entities)

        return entities

    @staticmethod
    def _merge_entity_lists(
        *lists: list[Any],
    ) -> list[Any]:
        seen: set[str] = set()
        merged: list[Any] = []
        for lst in lists:
            for entity in lst:
                key = (entity.text or "").upper().replace("-", "").replace(" ", "")
                if key not in seen:
                    seen.add(key)
                    merged.append(entity)
        return merged
