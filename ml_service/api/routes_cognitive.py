"""Cognitive memory routes — INT-1 fix.

Implements the two endpoints that .NET MLSearchService calls:
  POST /ml/index-document   — index text into cognitive memory
  POST /ml/semantic-search  — recall semantically similar documents

Graceful degradation: when cognitive memory is disabled or Weaviate is
not configured, both endpoints return empty/false responses instead of 500.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends

from .dependencies import verify_api_key
from .schemas import (
    IndexDocumentRequest,
    IndexDocumentResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
    SemanticSearchResultItem,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_cognitive_adapter():
    """Returns a CognitiveMemoryPort adapter if cognitive memory is enabled.

    Returns None when ML_ENABLE_COGNITIVE_MEMORY=False or Weaviate URL is
    not configured, allowing graceful degradation.
    """
    try:
        from ..config.loader import get_feature_flags
        flags = get_feature_flags()

        if not getattr(flags, "ML_ENABLE_COGNITIVE_MEMORY", False):
            return None

        weaviate_url = getattr(flags, "ML_COGNITIVE_MEMORY_URL", "") or ""
        if not weaviate_url.strip():
            return None

        from ...infrastructure.research.cognitive_memory_adapter import (
            WeaviateCognitiveAdapter,
        )
        return WeaviateCognitiveAdapter(url=weaviate_url)
    except Exception as exc:
        logger.warning("[COGNITIVE] Adapter unavailable: %s", exc)
        return None


@router.post("/ml/index-document", response_model=IndexDocumentResponse)
async def index_document(
    payload: IndexDocumentRequest,
    _: str = Depends(verify_api_key),
) -> IndexDocumentResponse:
    """Index a document into cognitive memory (Weaviate).

    Called by .NET MLSearchService.IndexDocumentAsync().
    Returns degraded response when cognitive memory is disabled.
    """
    adapter = _get_cognitive_adapter()

    if adapter is None:
        logger.debug(
            "[COGNITIVE] index-document skipped: cognitive memory disabled "
            "(tenant=%s source=%s)",
            payload.tenant_id,
            payload.source,
        )
        return IndexDocumentResponse(
            indexed=False,
            doc_id=None,
            chunk_count=0,
            reason="cognitive_memory_disabled",
        )

    try:
        doc_id = adapter.remember_document(
            text=payload.text,
            source=payload.source,
            classification=payload.classification,
            tenant_id=payload.tenant_id,
            analysis_result_id=payload.analysis_result_id,
        )
        logger.info(
            "[COGNITIVE] Indexed document doc_id=%s tenant=%s",
            doc_id,
            payload.tenant_id,
        )
        return IndexDocumentResponse(
            indexed=doc_id is not None,
            doc_id=doc_id,
            chunk_count=1,
        )
    except Exception as exc:
        logger.error(
            "[COGNITIVE] index-document failed: %s",
            exc,
            exc_info=True,
            extra={"tenant_id": payload.tenant_id, "source": payload.source},
        )
        return IndexDocumentResponse(
            indexed=False,
            doc_id=None,
            chunk_count=0,
            reason=f"indexing_error: {type(exc).__name__}",
        )


@router.post("/ml/semantic-search", response_model=SemanticSearchResponse)
async def semantic_search(
    payload: SemanticSearchRequest,
    _: str = Depends(verify_api_key),
) -> SemanticSearchResponse:
    """Recall semantically similar documents from cognitive memory.

    Called by .NET MLSearchService.SearchAsync().
    Returns empty results when cognitive memory is disabled.
    """
    adapter = _get_cognitive_adapter()

    if adapter is None:
        logger.debug(
            "[COGNITIVE] semantic-search skipped: cognitive memory disabled "
            "(tenant=%s query_len=%d)",
            payload.tenant_id,
            len(payload.query),
        )
        return SemanticSearchResponse(results=[], total=0)

    try:
        raw_results = adapter.recall_similar_documents(
            query=payload.query,
            tenant_id=payload.tenant_id,
            limit=payload.limit,
            domain=payload.domain,
        )
        items = [
            SemanticSearchResultItem(
                doc_id=r.get("doc_id", ""),
                content=r.get("content", ""),
                source=r.get("source", ""),
                classification=r.get("classification", ""),
                score=float(r.get("score", 0.0)),
                tenant_id=r.get("tenant_id", payload.tenant_id),
                analysis_result_id=r.get("analysis_result_id"),
            )
            for r in (raw_results or [])
        ]
        logger.info(
            "[COGNITIVE] semantic-search returned %d results tenant=%s",
            len(items),
            payload.tenant_id,
        )
        return SemanticSearchResponse(results=items, total=len(items))
    except Exception as exc:
        logger.error(
            "[COGNITIVE] semantic-search failed: %s",
            exc,
            exc_info=True,
            extra={"tenant_id": payload.tenant_id},
        )
        return SemanticSearchResponse(results=[], total=0)
