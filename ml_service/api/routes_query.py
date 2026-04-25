"""Chat / query route for the ML Service.

POST ``/ml/query`` — forwarded from ``.NET`` ``QueryCommandHandler``.
Thin orchestration: reuses the same text sub-analyzers and
``TextCognitiveEngine`` driven by the ingestion path.

Response contract expected by ``.NET``:
    ``{ "response_text": str, "metadata": object | null }``

Never returns HTTP 500 — ``.NET`` has no retry logic for 500s.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from iot_machine_learning.infrastructure.ml.cognitive.text import (
    TextAnalysisContext,
    TextAnalysisInput,
    TextCognitiveEngine,
)
from iot_machine_learning.infrastructure.ml.cognitive.text.analyzers import (
    compute_readability,
    compute_sentiment,
    compute_text_structure,
    compute_urgency,
)
from iot_machine_learning.infrastructure.ml.cognitive.text.text_pattern import (
    detect_text_patterns,
)

from .dependencies import verify_api_key
from .services.analyzers.text_recall import recall_similar_documents
from ..workers.result_writer import resolve_weaviate_url

logger = logging.getLogger(__name__)

# Dedicated engine for the query path — stateless per-call (see engine docstring);
# decoupled from the ingestion singleton to avoid import coupling.
_engine = TextCognitiveEngine()

router = APIRouter(prefix="/ml", tags=["query"])

_EMPTY_RESPONSE: Dict[str, Any] = {"response_text": "", "metadata": None}


class QueryRequest(BaseModel):
    """Payload produced by ``.NET`` ``QueryCommandHandler``."""

    session_id: str = Field(default="", description="Client-generated session id")
    message: str = Field(default="", description="User question / chat input")
    tenant_id: str = Field(default="", description="Tenant identifier")
    include_context: bool = Field(
        default=False,
        description="Reserved — client-side flag, currently ignored",
    )


class QueryResponse(BaseModel):
    """Minimum shape consumed by ``.NET`` ``QueryCommandHandler``."""

    response_text: str
    metadata: Optional[Dict[str, Any]] = None


def _build_input(message: str) -> TextAnalysisInput:
    """Mirror ``text_analyzer.analyze_text_document`` input assembly."""
    word_count = len(message.split())
    paragraph_count = max(1, message.count("\n\n") + 1)

    sentiment = compute_sentiment(message)
    urgency = compute_urgency(message)
    readability = compute_readability(message, word_count)
    structural = compute_text_structure(readability.sentences)
    patterns = detect_text_patterns(readability.sentences)

    return TextAnalysisInput(
        full_text=message,
        word_count=word_count,
        paragraph_count=paragraph_count,
        sentiment_score=sentiment.score,
        sentiment_label=sentiment.label,
        sentiment_positive_count=sentiment.positive_count,
        sentiment_negative_count=sentiment.negative_count,
        urgency_score=urgency.score,
        urgency_severity=urgency.severity,
        urgency_total_hits=urgency.total_hits,
        urgency_hits=urgency.hits,
        readability_avg_sentence_length=readability.avg_sentence_length,
        readability_n_sentences=readability.n_sentences,
        readability_vocabulary_richness=readability.vocabulary_richness,
        readability_embedded_numeric_count=readability.embedded_numeric_count,
        readability_sentences=readability.sentences,
        structural_regime=structural.regime,
        structural_trend=structural.trend,
        structural_stability=structural.stability,
        structural_noise=structural.noise,
        structural_available=structural.available,
        pattern_n_patterns=patterns.n_patterns,
        pattern_change_points=patterns.change_points,
        pattern_spikes=patterns.spikes,
        pattern_available=patterns.available,
        pattern_summary=patterns.summary,
    )


def _safe_recall(
    weaviate_url: Optional[str], message: str, tenant_id: str,
) -> List[Any]:
    """Best-effort semantic recall — always returns a list, never raises."""
    if not weaviate_url:
        return []
    try:
        return recall_similar_documents(
            weaviate_url,
            message,
            tenant_id=tenant_id,
            limit=5,
            min_certainty=0.6,
        )
    except Exception as exc:  # noqa: BLE001 — hard fail-safe boundary
        logger.warning("[ML-QUERY] recall_similar_documents failed: %s", exc)
        return []


@router.post("/query", response_model=QueryResponse)
async def ml_query(
    payload: QueryRequest,
    _: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Cognitive answer to a user chat message.

    Pipeline (graceful-fail at every step):
        1. Validate non-empty ``message``.
        2. Resolve Weaviate URL (may be ``None``).
        3. Semantic recall (best effort).
        4. Build ``TextAnalysisInput`` from the raw message.
        5. Call the ``TextCognitiveEngine``.
        6. Return ``{response_text, metadata}``.
    """
    session_id = payload.session_id or "no-session"
    message = (payload.message or "").strip()
    if not message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="message must be a non-empty string",
        )

    try:
        weaviate_url = resolve_weaviate_url()
        recall_results = _safe_recall(weaviate_url, message, payload.tenant_id)

        inp = _build_input(message)
        ctx = TextAnalysisContext(
            document_id=session_id,
            tenant_id=payload.tenant_id,
            filename="",
            weaviate_url=weaviate_url,
        )

        result = _engine.analyze(inp, ctx)

        metadata = result.to_dict()
        metadata["conclusion"] = result.conclusion
        metadata["n_recall_matches"] = len(recall_results)

        return {"response_text": result.conclusion, "metadata": metadata}

    except Exception as exc:  # noqa: BLE001 — never propagate 500 to .NET
        logger.warning(
            "[ML-QUERY] handler failed session_id=%s error=%s",
            session_id, exc,
        )
        return dict(_EMPTY_RESPONSE)
