"""Full text document analysis pipeline.

Orchestrates all text sub-analyzers (sentiment, urgency, readability,
structural) and assembles the final result dict with triggers,
thresholds, conclusion, and Explanation value object.

Single entry point: ``analyze_text_document(document_id, payload)``.
"""

from __future__ import annotations

from typing import Any, Dict

from .text_sentiment import compute_sentiment
from .text_urgency import compute_urgency
from .text_readability import compute_readability
from .text_structural import compute_text_structure
from .conclusion_builder import build_text_conclusion, build_text_explanation


def analyze_text_document(
    document_id: str, payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Run the full text analysis pipeline.

    Args:
        document_id: UUID of document.
        payload: Normalized payload with ``data.full_text``, etc.

    Returns:
        Result dict with ``analysis``, ``adaptive_thresholds``,
        ``conclusion``, ``confidence``.
    """
    data = payload.get("data", {})
    word_count = data.get("word_count", 0)
    paragraph_count = data.get("paragraph_count", 0)
    full_text = data.get("full_text", "")

    sentiment = compute_sentiment(full_text)
    urgency = compute_urgency(full_text)
    readability = compute_readability(full_text, word_count)
    structural = compute_text_structure(readability.sentences)

    # ── Analysis dict (preserves existing interface) ──
    analysis: Dict[str, Any] = {
        "sentiment": sentiment.label,
        "sentiment_score": sentiment.score,
        "urgency_score": urgency.score,
        "urgency_hits": urgency.hits,
        "readability": {
            "avg_sentence_length": readability.avg_sentence_length,
            "n_sentences": readability.n_sentences,
            "vocabulary_richness": readability.vocabulary_richness,
            "embedded_numeric_values": readability.embedded_numeric_count,
        },
        "structural": {
            "sentence_length_regime": structural.regime,
            "sentence_length_trend": structural.trend,
            "sentence_length_stability": structural.stability,
            "sentence_length_noise": structural.noise,
        } if structural.available else {},
        "triggers_activated": _build_triggers(sentiment, urgency),
    }

    # ── Confidence ──
    confidence = 0.75
    if word_count > 100:
        confidence = 0.80
    if word_count > 500:
        confidence = 0.85
    if structural.available:
        confidence += 0.05

    # ── Conclusion ──
    conclusion = build_text_conclusion(
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
    )

    # ── Structured explanation (Explanation value object) ──
    explanation_dict = build_text_explanation(
        document_id=document_id,
        sentiment_label=sentiment.label,
        sentiment_score=sentiment.score,
        urgency_score=urgency.score,
        urgency_severity=urgency.severity,
        urgency_hits=urgency.hits,
        readability_avg_sentence_len=readability.avg_sentence_length,
        readability_vocabulary_richness=readability.vocabulary_richness,
        structural_regime=structural.regime,
        structural_trend=structural.trend,
        structural_noise=structural.noise,
        confidence=confidence,
    )
    if explanation_dict:
        analysis["explanation"] = explanation_dict

    return {
        "analysis": analysis,
        "adaptive_thresholds": {
            "urgency_warning": 0.4,
            "urgency_critical": 0.7,
            "sentiment_negative": -0.2,
        },
        "conclusion": conclusion,
        "confidence": round(confidence, 3),
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
