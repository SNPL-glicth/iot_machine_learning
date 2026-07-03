"""Conclusion builders for text document analysis.

Funciones de formato puro — sin I/O, sin dependencias externas.
Toman los resultados de los sub-analyzers y producen texto legible.
"""

from __future__ import annotations

from typing import Any


def build_text_conclusion(
    sentiment_label: str = "neutral",
    urgency_severity: str = "info",
    structural_trend: str = "unknown",
    **_kwargs: Any,
) -> str:
    """Conclusion breve (legacy — no usado actualmente)."""
    return (
        f"Sentiment: {sentiment_label} | "
        f"Urgency: {urgency_severity} | "
        f"Trend: {structural_trend}"
    )


def build_text_explanation(
    sentiment_label: str = "neutral",
    urgency_score: float = 0.0,
    word_count: int = 0,
    **_kwargs: Any,
) -> str:
    """Explanation breve (legacy — no usado actualmente)."""
    return (
        f"Text analysis: {word_count} words, "
        f"{sentiment_label} sentiment, "
        f"urgency {urgency_score:.2f}"
    )


def build_semantic_conclusion(
    full_text: str,
    word_count: int,
    n_sentences: int,
    paragraph_count: int,
    sentiment_label: str,
    sentiment_score: float,
    urgency_score: float,
    urgency_total_hits: int,
    urgency_hits: list[str],
    urgency_severity: str,
    readability_avg_sentence_len: float,
    readability_vocabulary_richness: float,
    structural_regime: str,
    structural_trend: str,
    structural_available: bool,
    embedded_numeric_count: int,
    recall_results: list[Any],
    pattern_summary: Any,
) -> str:
    """Conclusion semántica completa — usada por text_analyzer.py.

    Args:
        full_text: Texto completo del documento.
        word_count: Conteo de palabras.
        n_sentences: Número de oraciones.
        paragraph_count: Número de párrafos.
        sentiment_label: "positive" | "negative" | "neutral".
        sentiment_score: Score de sentimiento [-1, 1].
        urgency_score: Score de urgencia [0, 1].
        urgency_total_hits: Total de keywords de urgencia.
        urgency_hits: Lista de keywords de urgencia encontradas.
        urgency_severity: "info" | "warning" | "critical".
        readability_avg_sentence_len: Longitud promedio de oración.
        readability_vocabulary_richness: Riqueza léxica (type-token ratio).
        structural_regime: Régimen estructural detectado.
        structural_trend: Tendencia estructural.
        structural_available: Si el análisis estructural está disponible.
        embedded_numeric_count: Cantidad de valores numéricos incrustados.
        recall_results: Resultados de búsqueda semántica similar.
        pattern_summary: Resumen de patrones detectados.

    Returns:
        Cadena de conclusión multilínea listo para presentación.
    """
    parts: list[str] = []

    severity_label = "info"
    if urgency_severity == "critical":
        severity_label = "critical"
    elif urgency_severity == "warning" or urgency_score > 0.4:
        severity_label = "warning"

    header = (
        f"Text incident — {severity_label.title()} | "
        f"Confidence: {_compute_confidence(sentiment_score, urgency_score):.1%}"
    )
    parts.append(header)

    analysis = (
        f"{word_count} words, {n_sentences} sentences, "
        f"{paragraph_count} paragraphs. "
        f"{sentiment_label.title()} sentiment ({sentiment_score:.2f}), "
        f"{urgency_severity} urgency ({urgency_score:.2f}, "
        f"{urgency_total_hits} hits)."
    )
    parts.append(analysis)

    if structural_available:
        parts.append(
            f"Structure: {structural_regime}, "
            f"trend {structural_trend}, "
            f"vocabulary richness {readability_vocabulary_richness:.2f}, "
            f"{embedded_numeric_count} numeric values."
        )

    if recall_results:
        parts.append(
            f"Semantic recall: {len(recall_results)} similar documents "
            f"(max score {max(r.score for r in recall_results):.3f})."
        )

    if pattern_summary:
        summary_str = str(pattern_summary)
        if len(summary_str) > 120:
            summary_str = summary_str[:117] + "..."
        parts.append(f"Patterns: {summary_str}")

    return "\n".join(parts)


def _compute_confidence(sentiment_score: float, urgency_score: float) -> float:
    """Confianza compuesta heuristicamente desde las senales disponibles."""
    base = 0.50
    urgency_conf = abs(urgency_score) * 0.25
    sent_conf = abs(sentiment_score) * 0.20
    return min(0.95, base + urgency_conf + sent_conf)
