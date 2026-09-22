"""Weaviate vector memory adapter implementing VectorMemoryPort."""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, List, Optional

from iot_machine_learning.domain.ports.vector_memory_port import VectorMemoryPort

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 10


class WeaviateVectorMemoryAdapter(VectorMemoryPort):
    """Adapter for querying Weaviate vector memory via GraphQL nearText."""

    def __init__(
        self,
        weaviate_url: Optional[str] = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self._url = weaviate_url or self._resolve_weaviate_url()
        self._timeout = timeout

    @staticmethod
    def _resolve_weaviate_url() -> Optional[str]:
        enabled = os.environ.get("WEAVIATE_ENABLED", "false").lower() == "true"
        if not enabled:
            return None
        return os.environ.get("WEAVIATE_URL", "http://localhost:8080").rstrip("/")

    def search_similar_explanations(
        self,
        concept: str,
        min_certainty: float,
        limit: int = 5,
    ) -> List[str]:
        """Search explanationText in Weaviate using nearText GraphQL query."""
        if not self._url:
            return []

        query = self._build_near_text_query(concept, min_certainty, limit)
        graphql_url = f"{self._url.rstrip('/')}/v1/graphql"
        body = json.dumps({"query": query}).encode("utf-8")
        req = urllib.request.Request(
            graphql_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
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

        explanations = []
        for item in items or []:
            expl_text = item.get("explanationText", "")
            if expl_text:
                explanations.append(expl_text)

        return explanations

    @staticmethod
    def _build_near_text_query(concept: str, certainty: float, limit: int) -> str:
        concept_escaped = json.dumps(concept)
        return (
            "{ Get { MLExplanation("
            f'nearText: {{ concepts: [{concept_escaped}], certainty: {certainty} }}, '
            f"limit: {limit}"
            ") { "
            "seriesId explanationText "
            "_additional { id certainty } "
            "} } }"
        )
