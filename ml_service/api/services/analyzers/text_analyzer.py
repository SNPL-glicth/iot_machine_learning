"""Full text document analysis pipeline.

Orchestrates all text sub-analyzers (sentiment, urgency, readability,
structural, chunking, embedding, semantic recall, pattern detection),
feeds pre-computed scores into ``TextCognitiveEngine`` for deep
cognitive reasoning, and assembles the final result dict.

The ml_service layer runs the analyzers and owns Weaviate I/O.
The engine (infrastructure layer) does cognitive reasoning only.

Single entry point: ``analyze_text_document(document_id, payload)``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .text_sentiment import compute_sentiment
from .text_urgency import compute_urgency
from .text_readability import compute_readability
from .text_structural import compute_text_structure
from .text_chunker import chunk_text
from .text_embedder import store_chunks
from .text_recall import recall_similar_documents, RecallResult
from .text_pattern import detect_text_patterns
from .conclusion_builder import (
    build_text_conclusion,
    build_text_explanation,
    build_semantic_conclusion,
)

from iot_machine_learning.infrastructure.ml.cognitive.text import (
    TextAnalysisContext,
    TextAnalysisInput,
    TextCognitiveEngine,
)

logger = logging.getLogger(__name__)

# Module-level engine instance (stateless per-call, safe to reuse)
_engine = TextCognitiveEngine()


def analyze_text_document(
    document_id: str, payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Run the full text analysis pipeline.

    Args:
        document_id: UUID of document.
        payload: Normalized payload with ``data.full_text``, etc.
            Optional keys for semantic enrichment:
            - ``_weaviate_url``: Weaviate base URL (enables embedding + recall).
            - ``_tenant_id``: Tenant identifier for isolation.
            - ``_analysis_id``: Reference to analysis_results row.

    Returns:
        Result dict with ``analysis``, ``adaptive_thresholds``,
        ``conclusion``, ``confidence``.
    """
    # DEBUG: Log payload structure
    logger.info(f"[TEXT_ANALYZER] analyze_text_document: document_id={document_id}, payload_keys={list(payload.keys())}")
    
    data = payload.get("data", {})
    logger.info(f"[TEXT_ANALYZER] Payload data keys: {list(data.keys())}")
    
    word_count = data.get("word_count", 0)
    paragraph_count = data.get("paragraph_count", 0)
    full_text = data.get("full_text", "")
    
    logger.info(f"[TEXT_ANALYZER] Extracted: word_count={word_count}, paragraph_count={paragraph_count}, full_text_length={len(full_text)}")
    logger.info(f"[TEXT_ANALYZER] Full text preview: {full_text[:100]!r}")
    
    if len(full_text.strip()) == 0:
        logger.warning(f"[TEXT_ANALYZER] Full text is empty or whitespace only!")
        # Return minimal result for empty text
        return {
            "analysis": {
                "sentiment": {"score": 0.0, "label": "neutral"},
                "urgency": {"score": 0.0, "level": "low"},
                "readability": {"score": 0.0, "level": "unknown"},
                "structural": {"complexity": "unknown"},
                "patterns": [],
                "data_points": 0,
                "domain": "general",
                "severity": "info",
                "confidence": 0.0,
            },
            "adaptive_thresholds": {},
            "conclusion": "No text content to analyze.",
            "confidence": 0.0,
        }

    # Semantic context (optional — passed by zenin_queue_poller)
    weaviate_url: Optional[str] = payload.get("_weaviate_url")
    tenant_id: str = str(payload.get("_tenant_id", ""))
    analysis_id: str = str(payload.get("_analysis_id", document_id))

    # ── Core analysis (always runs) ──
    sentiment = compute_sentiment(full_text)
    urgency = compute_urgency(full_text)
    readability = compute_readability(full_text, word_count)
    structural = compute_text_structure(readability.sentences)

    # ── Pattern detection on sentence signal ──
    patterns = detect_text_patterns(readability.sentences)

    # ── Semantic enrichment (graceful-fail if Weaviate is down) ──
    chunks = chunk_text(full_text)
    recall_results: List[RecallResult] = []

    if weaviate_url and chunks:
        # Recall FIRST (before storing current doc, to avoid self-match)
        recall_results = recall_similar_documents(
            weaviate_url,
            full_text,
            tenant_id=tenant_id,
            limit=3,
            min_certainty=0.7,
            exclude_analysis_id=analysis_id,
        )

    # ── Build TextAnalysisInput from pre-computed scores ──
    inp = TextAnalysisInput(
        full_text=full_text,
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

    ctx = TextAnalysisContext(
        document_id=document_id,
        tenant_id=tenant_id,
        filename=str(payload.get("_filename", "")),
        weaviate_url=weaviate_url,
    )

    # ── Run cognitive engine ──
    cognitive_result = _engine.analyze(inp, ctx)

    # ── Enrich conclusion with full semantic detail ──
    conclusion = build_semantic_conclusion(
        full_text=full_text,
        word_count=word_count,
        n_sentences=readability.n_sentences,
        paragraph_count=paragraph_count,
        sentiment_label=sentiment.label,
        sentiment_score=sentiment.score,
        urgency_score=urgency.score,
        urgency_total_hits=urgency.total_hits,
        urgency_hits=urgency.hits,
        urgency_severity=urgency.severity,
        readability_avg_sentence_len=readability.avg_sentence_length,
        readability_vocabulary_richness=readability.vocabulary_richness,
        structural_regime=structural.regime,
        structural_trend=structural.trend,
        structural_available=structural.available,
        embedded_numeric_count=readability.embedded_numeric_count,
        recall_results=recall_results,
        pattern_summary=patterns.summary,
    )

    # ── Merge cognitive analysis with existing fields ──
    analysis = cognitive_result.analysis
    analysis["triggers_activated"] = _build_triggers(sentiment, urgency)
    analysis["semantic"] = {
        "n_chunks": len(chunks),
        "n_recall_matches": len(recall_results),
        "recall_scores": [round(r.score, 3) for r in recall_results],
    } if chunks else {}

    # Add Explanation domain object to analysis
    analysis["explanation"] = cognitive_result.explanation.to_dict()

    # ── Store chunks AFTER conclusion is built (for future recall) ──
    if weaviate_url and chunks:
        chunk_ids = store_chunks(
            weaviate_url,
            chunks,
            tenant_id=tenant_id,
            analysis_id=analysis_id,
            filename=str(payload.get("_filename", "")),
            conclusion=conclusion,
        )
        if "semantic" in analysis and isinstance(analysis["semantic"], dict):
            analysis["semantic"]["n_chunks_stored"] = len(chunk_ids)

    return {
        "analysis": analysis,
        "adaptive_thresholds": {
            "urgency_warning": 0.4,
            "urgency_critical": 0.7,
            "sentiment_negative": -0.2,
        },
        "conclusion": conclusion,
        "confidence": round(cognitive_result.confidence, 3),
    }


def _build_triggers(sentiment, urgency):
    """Build trigger list from sentiment and urgency results."""
    triggers = []

    if urgency.severity == "critical":
        triggers.append({
            "type": "critical",
            "field": "urgency",
            "value": urgency.score,
            "threshold": 0.7,
            "message": "Urgencia alta detectada en el texto",
        })
    elif urgency.severity == "warning":
        triggers.append({
            "type": "warning",
            "field": "urgency",
            "value": urgency.score,
            "threshold": 0.4,
            "message": "Urgencia moderada detectada en el texto",
        })

    if sentiment.label == "negative" and sentiment.negative_count >= 3:
        triggers.append({
            "type": "warning",
            "field": "sentiment",
            "value": sentiment.score,
            "threshold": -0.2,
            "message": (
                f"Sentimiento negativo consistente "
                f"({sentiment.negative_count} indicadores)"
            ),
        })

    return triggers
