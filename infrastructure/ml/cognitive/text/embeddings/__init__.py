"""HybridEntityDetector — entity extraction with optional Weaviate server-side enrichment.

Design:
  ML_ENABLE_HYBRID_EMBEDDINGS=false (default):
    Pure regex passthrough — delegates to RegexEntityExtractor without
    duplicating its logic.  Zero new dependencies.

  ML_ENABLE_HYBRID_EMBEDDINGS=true:
    Uses Weaviate's server-side text2vec-transformers to retrieve
    semantically similar stored documents via nearText, then extracts
    entities from those documents (same RegexEntityExtractor on document
    text).  The result is the union of input-text entities and
    semantically-backed stored-document entities, deduplicated.
    magnitude_threshold controls min_certainty = 1 - threshold.

  On any error (Weaviate unavailable, timeout, no data): safe fallback
  to regex-only result.  Never raises, never returns None.

  No heavy ML deps — no sentence-transformers, no torch, consistent with
  the "zero heavy deps outside scikit-learn" rule.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from iot_machine_learning.infrastructure.ml.cognitive.text.semantic_extraction.composite_entity_extractor import (
    RegexEntityExtractor,
)

logger = logging.getLogger(__name__)

_WEAVIATE_TIMEOUT = 10
_WEAVIATE_QUERY_LIMIT = 5


@dataclass
class EntityResult:
    """Result container matching the original HybridEntityDetector contract.

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
    """Entity detector with hybrid regex + Weaviate server-side enrichment.

    Args:
        domain_hint: Domain context passed to RegexEntityExtractor.
        magnitude_threshold: Minimum semantic certainty for Weaviate-backed
            entities (converted to min_certainty = 1 - threshold).
    """

    def __init__(
        self,
        domain_hint: str = "general",
        magnitude_threshold: float = 0.3,
    ) -> None:
        self._domain = domain_hint
        self._threshold = magnitude_threshold
        self._regex = RegexEntityExtractor(domain_hint=domain_hint)

    # ── Public API ──────────────────────────────────────────────

    def extract_entities(self, text: str) -> EntityResult:
        """Extract entities using regex or hybrid regex+Weaviate.

        Steps:
          1. Always extract regex entities from the input text.
          2. If ML_ENABLE_HYBRID_EMBEDDINGS=false, return regex result.
          3. If true, query Weaviate nearText for semantically similar
             stored documents, extract entities from their explanationText,
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

        if not self._is_hybrid_enabled():
            return EntityResult(regex_entities)

        try:
            weaviate_entities = self._enrich_via_weaviate(text)
            if not weaviate_entities:
                return EntityResult(regex_entities)
            merged = self._merge_entity_lists(regex_entities, weaviate_entities)
            return EntityResult(merged)
        except Exception as exc:
            logger.debug("hybrid_weaviate_fallback: %s", exc)
            return EntityResult(regex_entities)

    # ── Internals ───────────────────────────────────────────────

    @staticmethod
    def _is_hybrid_enabled() -> bool:
        try:
            from iot_machine_learning.ml_service.config.feature_flags import (
                get_feature_flags,
            )

            flags = get_feature_flags()
            return bool(getattr(flags, "ML_ENABLE_HYBRID_EMBEDDINGS", False))
        except Exception:
            return False

    def _extract_regex(self, text: str) -> list[Any]:
        result = self._regex.extract(text)
        return list(result.entities)

    def _enrich_via_weaviate(self, text: str) -> list[Any]:
        url = self._resolve_weaviate_url()
        if not url:
            return []

        concept = " ".join(text.split()[:200])
        min_certainty = max(0.0, 1.0 - self._threshold)
        query = self._build_near_text_query(concept, min_certainty)

        graphql_url = f"{url.rstrip('/')}/v1/graphql"
        body = json.dumps({"query": query}).encode("utf-8")
        req = urllib.request.Request(
            graphql_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=_WEAVIATE_TIMEOUT) as resp:
                data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
            logger.debug("weaviate_query_failed: %s", exc)
            return []

        items: list[dict[str, Any]] = []
        try:
            items = data["data"]["Get"]["MLExplanation"]
        except (KeyError, TypeError):
            errors = data.get("errors", [])
            if errors:
                logger.debug("weaviate_graphql_errors: %s", [e.get("message", "") for e in errors[:3]])
            return []

        if not items:
            return []

        entities: list[Any] = []
        for item in items:
            expl_text = item.get("explanationText", "")
            if expl_text:
                result = self._regex.extract(expl_text)
                entities.extend(result.entities)

        return entities

    @staticmethod
    def _resolve_weaviate_url() -> Optional[str]:
        enabled = (
            os.environ.get("WEAVIATE_ENABLED", "false").lower() == "true"
        )
        if not enabled:
            return None
        url = os.environ.get("WEAVIATE_URL", "http://localhost:8080").rstrip("/")
        return url

    @staticmethod
    def _build_near_text_query(concept: str, certainty: float) -> str:
        concept_escaped = json.dumps(concept)
        return (
            "{ Get { MLExplanation("
            f'nearText: {{ concepts: [{concept_escaped}], certainty: {certainty} }}, '
            f"limit: {_WEAVIATE_QUERY_LIMIT}"
            ") { "
            "seriesId explanationText "
            "_additional { id certainty } "
            "} } }"
        )

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
