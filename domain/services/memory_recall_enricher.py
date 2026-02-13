"""Domain service: enrich predictions with recalled cognitive memory.

Pure domain logic — no infrastructure dependencies.  Receives a
``CognitiveMemoryPort`` (abstraction) and uses it to query for
similar past explanations and anomalies.

The enricher ONLY adds contextual text.  It never modifies:
    - Predicted numeric value
    - Anomaly boolean
    - Confidence score
    - Threshold logic

Design:
    - Fail-safe: if any recall call fails, returns empty enrichment.
    - Zero latency when disabled (short-circuits immediately).
    - Returns a plain ``MemoryRecallContext`` dataclass consumed by
      the use case to populate the DTO.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..entities.memory_search_result import MemorySearchResult
from ..entities.prediction import Prediction
from ..ports.cognitive_memory_port import CognitiveMemoryPort

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryRecallContext:
    """Enrichment data produced by memory recall.

    Consumed by the use case to populate ``PredictionDTO.memory_context``.
    Contains only textual/contextual additions — no numeric overrides.

    Attributes:
        similar_explanations: Top-k similar past explanations.
        similar_anomalies: Top-k similar past anomaly events.
        enriched_explanation: Human-readable summary combining
            the original explanation with historical references.
        historical_references: Short reference strings like
            ``"Similar to event on 2025-02-01 (certainty=0.91)"``
            for UI display.
    """

    similar_explanations: List[MemorySearchResult] = field(default_factory=list)
    similar_anomalies: List[MemorySearchResult] = field(default_factory=list)
    enriched_explanation: str = ""
    historical_references: List[str] = field(default_factory=list)

    @property
    def has_context(self) -> bool:
        """``True`` if any memory was recalled."""
        return bool(self.similar_explanations or self.similar_anomalies)

    def to_dict(self) -> Dict[str, object]:
        """Serialize for API responses or logging."""
        result: Dict[str, object] = {}
        if self.enriched_explanation:
            result["enriched_explanation"] = self.enriched_explanation
        if self.historical_references:
            result["historical_references"] = self.historical_references
        if self.similar_explanations:
            result["similar_explanations"] = [
                r.to_dict() for r in self.similar_explanations
            ]
        if self.similar_anomalies:
            result["similar_anomalies"] = [
                r.to_dict() for r in self.similar_anomalies
            ]
        return result


_EMPTY_CONTEXT = MemoryRecallContext()


class MemoryRecallEnricher:
    """Enriches a prediction with recalled cognitive memory.

    Args:
        cognitive: A ``CognitiveMemoryPort`` implementation.
        top_k: Maximum number of similar memories to retrieve.
        min_certainty: Minimum semantic similarity threshold.
    """

    def __init__(
        self,
        cognitive: CognitiveMemoryPort,
        *,
        top_k: int = 3,
        min_certainty: float = 0.7,
    ) -> None:
        self._cognitive = cognitive
        self._top_k = top_k
        self._min_certainty = min_certainty

    def enrich(self, prediction: Prediction) -> MemoryRecallContext:
        """Query cognitive memory and build enrichment context.

        Never raises.  Returns empty ``MemoryRecallContext`` on any error.

        Args:
            prediction: The prediction to enrich with historical context.

        Returns:
            ``MemoryRecallContext`` with recalled memories and
            enriched explanation text.
        """
        try:
            return self._do_enrich(prediction)
        except Exception as exc:
            logger.warning(
                "memory_recall_enrichment_failed",
                extra={
                    "series_id": prediction.series_id,
                    "error": str(exc),
                },
            )
            return _EMPTY_CONTEXT

    def _do_enrich(self, prediction: Prediction) -> MemoryRecallContext:
        """Internal enrichment logic."""
        query_text = self._build_query(prediction)
        if not query_text:
            return _EMPTY_CONTEXT

        # Recall similar explanations
        similar_explanations = self._cognitive.recall_similar_explanations(
            query_text,
            series_id=prediction.series_id,
            engine_name=prediction.engine_name,
            limit=self._top_k,
            min_certainty=self._min_certainty,
        )

        # Recall similar anomalies (using same query text)
        similar_anomalies = self._cognitive.recall_similar_anomalies(
            query_text,
            series_id=prediction.series_id,
            limit=self._top_k,
            min_certainty=self._min_certainty,
        )

        if not similar_explanations and not similar_anomalies:
            return _EMPTY_CONTEXT

        # Build historical references
        historical_refs = self._build_references(
            similar_explanations, similar_anomalies
        )

        # Build enriched explanation
        original_explanation = str(
            prediction.metadata.get("explanation", "")
        )
        enriched = self._build_enriched_explanation(
            original_explanation, historical_refs
        )

        return MemoryRecallContext(
            similar_explanations=similar_explanations,
            similar_anomalies=similar_anomalies,
            enriched_explanation=enriched,
            historical_references=historical_refs,
        )

    def _build_query(self, prediction: Prediction) -> str:
        """Build a semantic search query from the prediction."""
        parts: List[str] = []

        explanation = prediction.metadata.get("explanation", "")
        if explanation:
            parts.append(str(explanation))

        parts.append(
            f"series {prediction.series_id} trend {prediction.trend} "
            f"confidence {prediction.confidence_score:.2f}"
        )

        return " ".join(parts)

    def _build_references(
        self,
        explanations: List[MemorySearchResult],
        anomalies: List[MemorySearchResult],
    ) -> List[str]:
        """Build short human-readable reference strings."""
        refs: List[str] = []

        for mem in explanations:
            date_str = mem.created_at or "unknown date"
            refs.append(
                f"Similar prediction on {date_str} "
                f"(certainty={mem.certainty:.2f})"
            )

        for mem in anomalies:
            date_str = mem.created_at or "unknown date"
            refs.append(
                f"Similar anomaly on {date_str} "
                f"(certainty={mem.certainty:.2f})"
            )

        return refs

    def _build_enriched_explanation(
        self,
        original: str,
        references: List[str],
    ) -> str:
        """Combine original explanation with historical references."""
        if not references:
            return original

        ref_text = "; ".join(references)

        if original:
            return f"{original} | Historical context: {ref_text}"
        return f"Historical context: {ref_text}"
