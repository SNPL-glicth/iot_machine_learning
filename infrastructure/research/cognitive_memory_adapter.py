"""Cognitive memory adapter for document indexing and semantic search.

Wraps Weaviate REST API (raw HTTP, no SDK) for the ZENIN document domain.
Used by routes_cognitive.py to implement INT-1 endpoints:
  - remember_document()        → POST /ml/index-document
  - recall_similar_documents() → POST /ml/semantic-search

Fail-safe: every method returns None/[] on failure — never raises.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 10
_WEAVIATE_CLASS = "ZeninDocument"


class WeaviateCognitiveAdapter:
    """Thin HTTP adapter for document cognitive memory (Weaviate).

    Args:
        url: Weaviate REST base URL (e.g. ``http://localhost:8080``).
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

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def remember_document(
        self,
        text: str,
        source: str,
        classification: str,
        tenant_id: str,
        analysis_result_id: Optional[str] = None,
    ) -> Optional[str]:
        """Index a document chunk into Weaviate.

        Returns the Weaviate object ID or None on failure.
        """
        doc_id = str(uuid.uuid4())
        payload: Dict[str, Any] = {
            "class": _WEAVIATE_CLASS,
            "id": doc_id,
            "properties": {
                "content": text,
                "source": source,
                "classification": classification,
                "tenantId": tenant_id,
                "analysisResultId": analysis_result_id or "",
            },
        }

        if self._dry_run:
            logger.debug(
                "[COGNITIVE] dry_run remember_document doc_id=%s tenant=%s",
                doc_id,
                tenant_id,
            )
            return doc_id

        try:
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self._base_url}/v1/objects",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("id", doc_id)
        except urllib.error.HTTPError as exc:
            logger.error(
                "[COGNITIVE] remember_document HTTP %d: %s",
                exc.code,
                exc.reason,
            )
        except Exception as exc:
            logger.error("[COGNITIVE] remember_document failed: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def recall_similar_documents(
        self,
        query: str,
        tenant_id: str,
        limit: int = 5,
        domain: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for semantically similar documents in Weaviate.

        Returns a list of result dicts with keys:
            doc_id, content, source, classification, score, tenant_id,
            analysis_result_id
        Empty list on failure or when no results.
        """
        where_filter: Optional[Dict[str, Any]] = {
            "path": ["tenantId"],
            "operator": "Equal",
            "valueString": tenant_id,
        }

        graphql_query = self._build_graphql_query(
            query=query,
            limit=limit,
            where_filter=where_filter,
        )

        if self._dry_run:
            logger.debug(
                "[COGNITIVE] dry_run recall_similar_documents tenant=%s limit=%d",
                tenant_id,
                limit,
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_graphql_query(
        self,
        query: str,
        limit: int,
        where_filter: Optional[Dict[str, Any]],
    ) -> str:
        where_clause = ""
        if where_filter:
            path = json.dumps(where_filter["path"])
            op = where_filter["operator"]
            val = json.dumps(where_filter.get("valueString", ""))
            where_clause = (
                f', where: {{path: {path}, operator: {op}, valueString: {val}}}'
            )

        return f"""
        {{
            Get {{
                {_WEAVIATE_CLASS}(
                    nearText: {{concepts: [{json.dumps(query)}]}}
                    limit: {limit}
                    {where_clause}
                ) {{
                    content
                    source
                    classification
                    tenantId
                    analysisResultId
                    _additional {{
                        id
                        certainty
                    }}
                }}
            }}
        }}
        """

    @staticmethod
    def _parse_graphql_results(
        data: Dict[str, Any],
        tenant_id: str,
    ) -> List[Dict[str, Any]]:
        try:
            items = (
                data.get("data", {})
                .get("Get", {})
                .get(_WEAVIATE_CLASS, [])
            )
        except Exception:
            return []

        results = []
        for item in items or []:
            additional = item.get("_additional", {})
            results.append(
                {
                    "doc_id": additional.get("id", ""),
                    "content": item.get("content", ""),
                    "source": item.get("source", ""),
                    "classification": item.get("classification", ""),
                    "score": float(additional.get("certainty", 0.0)),
                    "tenant_id": item.get("tenantId", tenant_id),
                    "analysis_result_id": item.get("analysisResultId") or None,
                }
            )
        return results
