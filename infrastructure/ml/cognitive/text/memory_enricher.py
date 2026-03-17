"""TextMemoryEnricher — recalls similar documents from cognitive memory.

Uses ``CognitiveMemoryPort`` to query for semantically similar past
explanations and patterns, then builds enrichment context with
historical references.

Graceful-fail: returns empty context if memory port is unavailable
or any recall call fails.

No imports from ml_service — only domain layer.
Single entry point: ``TextMemoryEnricher.enrich()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from iot_machine_learning.domain.entities.memory_search_result import (
    MemorySearchResult,
)

logger = logging.getLogger(__name__)

# Maximum words from the document used to build the recall query
_QUERY_MAX_WORDS = 200
_DEFAULT_LIMIT = 3
_DEFAULT_MIN_CERTAINTY = 0.7


@dataclass(frozen=True)
class TextRecallContext:
    """Memory recall enrichment for text analysis.

    Attributes:
        similar_explanations: Top-k similar past explanations.
        similar_patterns: Top-k similar past patterns.
        enriched_summary: Human-readable summary combining
            historical references.
        historical_references: Short reference strings for display.
    """

    similar_explanations: List[MemorySearchResult] = field(default_factory=list)
    similar_patterns: List[MemorySearchResult] = field(default_factory=list)
    enriched_summary: str = ""
    historical_references: List[str] = field(default_factory=list)

    @property
    def has_context(self) -> bool:
        """``True`` if any memory was recalled."""
        return bool(self.similar_explanations or self.similar_patterns)

    def to_dict(self) -> Dict[str, object]:
        """Serialize for API responses."""
        result: Dict[str, object] = {}
        if self.enriched_summary:
            result["enriched_summary"] = self.enriched_summary
        if self.historical_references:
            result["historical_references"] = self.historical_references
        if self.similar_explanations:
            result["similar_explanations"] = [
                r.to_dict() for r in self.similar_explanations
            ]
        if self.similar_patterns:
            result["similar_patterns"] = [
                r.to_dict() for r in self.similar_patterns
            ]
        return result


_EMPTY = TextRecallContext()


class TextMemoryEnricher:
    """Enriches text analysis with recalled cognitive memory.

    Stateless — safe to reuse across documents.
    """

    def enrich(
        self,
        full_text: str,
        domain: str,
        cognitive_memory: Optional[object],
        document_id: str = "",
    ) -> TextRecallContext:
        """Query cognitive memory for similar past documents.

        Args:
            full_text: Full document text.
            domain: Classified document domain.
            cognitive_memory: ``CognitiveMemoryPort`` implementation,
                or ``None`` to skip recall.
            document_id: Current document ID (used as series_id filter).

        Returns:
            ``TextRecallContext`` with recalled memories.
            Empty if memory port is unavailable or recall fails.
        """
        if cognitive_memory is None:
            return _EMPTY

        try:
            return self._do_enrich(full_text, domain, cognitive_memory, document_id)
        except Exception as exc:
            logger.warning("[TEXT_MEMORY] Recall failed: %s", exc)
            return _EMPTY

    def _do_enrich(
        self,
        full_text: str,
        domain: str,
        memory_port: object,
        document_id: str,
    ) -> TextRecallContext:
        """Internal enrichment logic."""
        query = self._build_query(full_text, domain)
        if not query:
            return _EMPTY

        similar_explanations: List[MemorySearchResult] = []
        similar_patterns: List[MemorySearchResult] = []

        # Recall similar explanations
        if hasattr(memory_port, "recall_similar_explanations"):
            similar_explanations = memory_port.recall_similar_explanations(
                query,
                series_id=document_id or None,
                limit=_DEFAULT_LIMIT,
                min_certainty=_DEFAULT_MIN_CERTAINTY,
            )

        # Recall similar patterns
        if hasattr(memory_port, "recall_similar_patterns"):
            similar_patterns = memory_port.recall_similar_patterns(
                query,
                series_id=document_id or None,
                limit=_DEFAULT_LIMIT,
                min_certainty=_DEFAULT_MIN_CERTAINTY,
            )

        if not similar_explanations and not similar_patterns:
            return _EMPTY

        references = self._build_references(similar_explanations, similar_patterns)
        summary = self._build_summary(references)

        return TextRecallContext(
            similar_explanations=similar_explanations,
            similar_patterns=similar_patterns,
            enriched_summary=summary,
            historical_references=references,
        )

    def _build_query(self, full_text: str, domain: str) -> str:
        """Build a semantic search query from the document."""
        words = full_text.split()[:_QUERY_MAX_WORDS]
        query = " ".join(words)
        if domain and domain != "general":
            query = f"{domain} analysis: {query}"
        return query

    def _build_references(
        self,
        explanations: List[MemorySearchResult],
        patterns: List[MemorySearchResult],
    ) -> List[str]:
        """Build short human-readable reference strings."""
        refs: List[str] = []

        for mem in explanations:
            date_str = mem.created_at or "unknown date"
            refs.append(
                f"Similar document on {date_str} "
                f"(certainty={mem.certainty:.2f})"
            )

        for mem in patterns:
            date_str = mem.created_at or "unknown date"
            refs.append(
                f"Similar pattern on {date_str} "
                f"(certainty={mem.certainty:.2f})"
            )

        return refs

    def _build_summary(self, references: List[str]) -> str:
        """Combine references into enriched summary."""
        if not references:
            return ""
        return "Historical context: " + "; ".join(references)
