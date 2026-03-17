"""Semantic recall of similar past documents via Weaviate nearText.

Queries the existing ``MLExplanation`` objects (domainName="zenin_docs")
using Weaviate's text2vec-transformers vectorization to find documents
that are semantically similar to the current one.

Graceful-fail: if Weaviate is unreachable, returns an empty list.
The analysis pipeline continues with heuristic-only conclusions.

Single entry point: ``recall_similar_documents(weaviate_url, ...)``.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TIMEOUT = 10


@dataclass(frozen=True)
class RecallResult:
    """A semantically similar past document found by vector search.

    Attributes:
        doc_id: Weaviate object UUID.
        content: The matched text (chunk or explanation).
        score: Semantic similarity score (0.0–1.0).
        filename: Original filename of the matched document.
        conclusion: Conclusion from the matched document's analysis.
        metadata: Additional properties from the matched object.
    """

    doc_id: str
    content: str
    score: float
    filename: str = ""
    conclusion: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "content": self.content[:300],
            "score": self.score,
            "filename": self.filename,
            "conclusion": self.conclusion,
        }


def recall_similar_documents(
    weaviate_url: Optional[str],
    query_text: str,
    *,
    tenant_id: str = "",
    limit: int = 3,
    min_certainty: float = 0.7,
    exclude_analysis_id: str = "",
) -> List[RecallResult]:
    """Find semantically similar past documents in Weaviate.

    Uses ``nearText`` on the ``MLExplanation`` class filtered to
    ``domainName="zenin_docs"``.

    Args:
        weaviate_url: Weaviate base URL.  If ``None``, returns [].
        query_text: Text to search by (first ~200 words used as query).
        tenant_id: Filter results to this tenant (empty = all).
        limit: Max results to return.
        min_certainty: Minimum similarity threshold.
        exclude_analysis_id: Skip results matching this analysis_id
            (to avoid self-matching).

    Returns:
        List of ``RecallResult`` ordered by similarity descending.
    """
    if not weaviate_url or not query_text or not query_text.strip():
        return []

    graphql_url = f"{weaviate_url.rstrip('/')}/v1/graphql"

    # Use first ~200 words as the query concept
    query_words = query_text.split()[:200]
    concept = " ".join(query_words)

    query = _build_graphql_query(
        concept=concept,
        limit=limit,
        certainty=min_certainty,
    )

    try:
        body = json.dumps({"query": query}).encode("utf-8")
        req = urllib.request.Request(
            graphql_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results = _parse_response(data, exclude_analysis_id, tenant_id)
        if results:
            logger.info(
                "[TEXT_RECALL] Found %d similar documents (top score=%.3f)",
                len(results), results[0].score,
            )
        return results

    except Exception as exc:
        logger.warning("[TEXT_RECALL] Semantic recall failed: %s", exc)
        return []


def _build_graphql_query(
    *,
    concept: str,
    limit: int,
    certainty: float,
) -> str:
    """Build the GraphQL nearText query for zenin_docs."""
    concept_escaped = json.dumps(concept)
    where_filter = json.dumps({
        "path": ["domainName"],
        "operator": "Equal",
        "valueText": "zenin_docs",
    })
    return (
        "{ Get { MLExplanation("
        f'nearText: {{ concepts: [{concept_escaped}], certainty: {certainty} }}, '
        f"where: {where_filter}, "
        f"limit: {limit}"
        ") { "
        "seriesId explanationText auditTraceId metadata "
        "_additional { id certainty } "
        "} } }"
    )


def _parse_response(
    data: Dict[str, Any],
    exclude_analysis_id: str,
    tenant_id: str,
) -> List[RecallResult]:
    """Parse Weaviate GraphQL response into RecallResult list."""
    try:
        items = data["data"]["Get"]["MLExplanation"]
    except (KeyError, TypeError):
        errors = data.get("errors", [])
        if errors:
            logger.warning(
                "[TEXT_RECALL] GraphQL errors: %s",
                [e.get("message", "") for e in errors[:3]],
            )
        return []

    if not items:
        return []

    results: List[RecallResult] = []
    for item in items:
        additional = item.get("_additional", {})
        doc_id = additional.get("id", "")
        certainty = float(additional.get("certainty", 0.0))

        # Parse metadata JSON
        meta_raw = item.get("metadata", "")
        meta: Dict[str, Any] = {}
        if meta_raw:
            try:
                meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
            except (json.JSONDecodeError, TypeError):
                meta = {}

        # Filter: skip self-match
        audit_id = item.get("auditTraceId", "")
        if exclude_analysis_id and audit_id == exclude_analysis_id:
            continue

        # Filter: tenant isolation
        if tenant_id and meta.get("tenant_id", "") != tenant_id:
            continue

        results.append(RecallResult(
            doc_id=doc_id,
            content=item.get("explanationText", ""),
            score=certainty,
            filename=meta.get("filename", item.get("seriesId", "")),
            conclusion=meta.get("conclusion", ""),
            metadata=meta,
        ))

    return results
