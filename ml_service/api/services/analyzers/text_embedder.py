"""Weaviate text2vec chunk storage.

Stores document chunks in Weaviate using the existing ``MLExplanation``
class with ``text2vec-transformers`` vectorization.  Uses raw HTTP —
no SDK dependency.

Graceful-fail: if Weaviate is unreachable, returns an empty list and
logs a warning.  The analysis pipeline continues with heuristic-only
conclusions.

Single entry point: ``store_chunks(weaviate_url, ...)``.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
import uuid as _uuid
from typing import Any, Dict, List, Optional

from .text_chunker import TextChunk

logger = logging.getLogger(__name__)

_TIMEOUT = 10


def store_chunks(
    weaviate_url: Optional[str],
    chunks: List[TextChunk],
    *,
    tenant_id: str,
    analysis_id: str,
    filename: str = "",
    conclusion: str = "",
) -> List[str]:
    """Store text chunks in Weaviate for later semantic recall.

    Each chunk is stored as an ``MLExplanation`` object with
    ``domainName="zenin_docs"`` so text2vec-transformers auto-vectorizes
    the ``explanationText`` field.

    Args:
        weaviate_url: Weaviate base URL (e.g. ``http://localhost:8080``).
            If ``None``, storage is skipped silently.
        chunks: List of ``TextChunk`` from the chunker.
        tenant_id: Tenant identifier for isolation.
        analysis_id: Reference to ``zenin_docs.analysis_results.Id``.
        filename: Original filename (for metadata).
        conclusion: Document-level conclusion (stored in metadata for
            recall enrichment).

    Returns:
        List of Weaviate object UUIDs (one per stored chunk).
        Empty list if Weaviate is unavailable or *weaviate_url* is None.
    """
    if not weaviate_url or not chunks:
        return []

    objects_url = f"{weaviate_url.rstrip('/')}/v1/objects"
    stored_ids: List[str] = []

    for chunk in chunks:
        doc_id = _store_single_chunk(
            objects_url,
            chunk=chunk,
            tenant_id=tenant_id,
            analysis_id=analysis_id,
            filename=filename,
            conclusion=conclusion,
        )
        if doc_id:
            stored_ids.append(doc_id)

    if stored_ids:
        logger.info(
            "[TEXT_EMBEDDER] Stored %d/%d chunks for %s (analysis=%s)",
            len(stored_ids), len(chunks), filename, analysis_id,
        )
    return stored_ids


def _store_single_chunk(
    objects_url: str,
    *,
    chunk: TextChunk,
    tenant_id: str,
    analysis_id: str,
    filename: str,
    conclusion: str,
) -> Optional[str]:
    """Store one chunk as an MLExplanation object. Returns UUID or None."""
    doc_id = str(_uuid.uuid4())

    payload = json.dumps({
        "class": "MLExplanation",
        "id": doc_id,
        "properties": {
            "domainName": "zenin_docs",
            "seriesId": filename,
            "engineName": "document_analyzer",
            "explanationText": chunk.text[:2000],
            "trend": "neutral",
            "confidenceScore": 0.0,
            "confidenceLevel": "medium",
            "predictedValue": 0.0,
            "horizonSteps": 0,
            "featureContributions": json.dumps({
                "chunk_index": chunk.index,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "token_estimate": chunk.token_estimate,
            }),
            "auditTraceId": analysis_id,
            "metadata": json.dumps({
                "tenant_id": tenant_id,
                "analysis_id": analysis_id,
                "filename": filename,
                "chunk_index": chunk.index,
                "conclusion": conclusion[:500] if conclusion else "",
            }),
        },
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            objects_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            if resp.status in (200, 201):
                return doc_id
        return None
    except Exception as exc:
        logger.warning(
            "[TEXT_EMBEDDER] Failed to store chunk %d: %s",
            chunk.index, exc,
        )
        return None
