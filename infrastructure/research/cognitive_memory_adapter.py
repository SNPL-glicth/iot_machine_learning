"""Cognitive memory adapter for document indexing and semantic search.

Wraps Weaviate REST API (raw HTTP, no SDK) for the ZENIN document domain.
Used by routes_cognitive.py to implement INT-1 endpoints.
Fail-safe: every method returns None/[] on failure — never raises.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, List, Optional

from .weaviate_graphql import build_graphql_query, parse_graphql_results
from .weaviate_writer import index_document

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 10
_WEAVIATE_CLASS = "ZeninDocument"


class WeaviateCognitiveAdapter:
    """Thin HTTP adapter for document cognitive memory (Weaviate).

    Args:
        url: Weaviate REST base URL.
        dry_run: Log payloads without sending to Weaviate.
        timeout: HTTP timeout in seconds.
    """

    def __init__(
        self,
        url: str = "http://localhost:8080",
        *,
        dry_run: bool = False,
        timeout: int = _TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = url.rstrip("/")
        self._dry_run = dry_run
        self._timeout = timeout

    def remember_document(
        self,
        text: str,
        source: str,
        classification: str,
        tenant_id: str,
        analysis_result_id: Optional[str] = None,
        context_type: str = "documental",
    ) -> Optional[str]:
        """Index a document chunk into Weaviate."""
        return index_document(
            base_url=self._base_url,
            timeout=self._timeout,
            dry_run=self._dry_run,
            text=text,
            source=source,
            classification=classification,
            tenant_id=tenant_id,
            analysis_result_id=analysis_result_id,
            context_type=context_type,
        )

    def recall_similar_documents(
        self,
        query: str,
        tenant_id: str,
        limit: int = 5,
        domain: Optional[str] = None,
        context_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for semantically similar documents in Weaviate.

        Args:
            query: Search query text.
            tenant_id: Tenant identifier.
            limit: Maximum number of results.
            domain: Optional domain filter.
            context_type: Optional data context filter ("numeric", "documental").
                Filters out cross-context memories to prevent semantic feedback loops.

        Returns:
            A list of result dicts with keys:
                doc_id, content, source, classification, score, tenant_id,
                analysis_result_id, context_type
            Empty list on failure or when no results.
        """
        # Build compound where filter: tenant AND optional context_type
        where_operands: List[Dict[str, Any]] = [
            {
                "path": ["tenantId"],
                "operator": "Equal",
                "valueString": tenant_id,
            },
        ]
        if context_type is not None:
            where_operands.append({
                "path": ["contextType"],
                "operator": "Equal",
                "valueString": context_type,
            })

        where_filter: Optional[Dict[str, Any]] = {
            "operator": "And",
            "operands": where_operands,
        }

        graphql_query = self._build_graphql_query(
            query=query,
            limit=limit,
            where_filter=where_filter,
        )

        if self._dry_run:
            logger.debug(
                "[COGNITIVE] dry_run recall_similar_documents tenant=%s limit=%d context=%s",
                tenant_id,
                limit,
                context_type,
            )
            return []

        try:
            body = json.dumps({"query": graphql_query}).encode("utf-8")
            req = urllib.request.Request(
                f"{self._base_url}/v1/graphql",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return self._parse_graphql_results(data, tenant_id)
        except urllib.error.HTTPError as exc:
            logger.error(
                "[COGNITIVE] recall_similar_documents HTTP %d: %s",
                exc.code,
                exc.reason,
            )
        except Exception as exc:
            logger.error("[COGNITIVE] recall_similar_documents failed: %s", exc)
        return []

    def _build_graphql_query(
        self,
        query: str,
        limit: int,
        where_filter: Optional[Dict[str, Any]],
    ) -> str:
        return build_graphql_query(query, limit, where_filter)

    @staticmethod
    def _parse_graphql_results(
        data: Dict[str, Any],
        tenant_id: str,
    ) -> List[Dict[str, Any]]:
        return parse_graphql_results(data, tenant_id)
