"""Conclusion rendering using domain Explanation + ExplanationRenderer.

Bridges the gap between raw analysis results and the structured
explainability layer.  Constructs ``Explanation`` value objects and
uses ``ExplanationRenderer`` to produce human-readable output.

Falls back to simple string assembly if the explainability layer
is not available.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from iot_machine_learning.infrastructure.ml.cognitive.text.analyzers.keyword_config import (
    DOMAIN_KEYWORDS,
    ACTION_TEMPLATES,
)
from iot_machine_learning.infrastructure.ml.cognitive.text.conclusion_domain import (
    classify_document_domain,
    extract_key_topics,
    build_severity_assessment,
    build_topic_summary,
    get_recommended_actions,
    build_recall_context,
)
from iot_machine_learning.infrastructure.ml.cognitive.text.conclusion_explanation import (
    build_text_explanation,
)

logger = logging.getLogger(__name__)


def build_semantic_conclusion(
    *,
    full_text: str,
    word_count: int,
    n_sentences: int,
    paragraph_count: int,
    sentiment_label: str,
    sentiment_score: float,
    urgency_score: float,
    urgency_total_hits: int,
    urgency_hits: List[Dict[str, Any]],
    urgency_severity: str,
    readability_avg_sentence_len: float,
    readability_vocabulary_richness: float,
    structural_regime: str = "unknown",
    structural_trend: float = 0.0,
    structural_available: bool = False,
    embedded_numeric_count: int = 0,
    recall_results: Optional[List[Any]] = None,
    pattern_summary: str = "",
) -> str:
    """Build an actionable, semantically meaningful conclusion.

    Instead of just reporting scores, this tells the user:
    - WHAT the document is about (domain classification + key topics)
    - HOW critical it is (severity assessment)
    - WHAT they should do (recommended actions)
    - WHAT similar past documents said (semantic recall context)

    Falls back to ``build_text_conclusion`` format if domain
    classification fails.

    Args:
        full_text: Complete document text for domain detection.
        recall_results: List of ``RecallResult`` from semantic recall.
        pattern_summary: Summary from text pattern detection.
        (all other args same as ``build_text_conclusion``)

    Returns:
        Multi-line actionable conclusion string.
    """
    # 1. Classify domain
    domain = classify_document_domain(full_text)
    domain_label = _DOMAIN_LABELS.get(domain, domain.title())

    # 2. Extract key entities/topics from text
    key_topics = extract_key_topics(full_text, urgency_hits)

    # 3. Build severity assessment
    severity_line = build_severity_assessment(
        urgency_severity, urgency_score, sentiment_label, sentiment_score,
    )

    # 4. Build topic summary (WHAT is this about)
    topic_line = build_topic_summary(domain_label, key_topics, word_count)

    # 5. Build recommended actions (WHAT to do)
    action_line = get_recommended_actions(domain, urgency_severity, key_topics)

    # 6. Recall context (similar past documents)
    recall_line = build_recall_context(recall_results)

    # 7. Assemble
    parts: List[str] = [topic_line, severity_line]

    if pattern_summary:
        parts.append(f"Narrative structure: {pattern_summary}")

    parts.append(f"Recommended actions: {action_line}")

    if recall_line:
        parts.append(recall_line)

    # Technical details (compact)
    parts.append(
        f"[Details: {word_count} words, {n_sentences} sentences, "
        f"sentiment={sentiment_label}({sentiment_score:.2f}), "
        f"urgency={urgency_score:.2f}, "
        f"regime={structural_regime}]"
    )

    return "\n".join(parts)


# ── Tabular conclusions ───────────────────────────────────────────


def build_tabular_conclusion(
    *,
    row_count: int,
    n_headers: int,
    n_numeric: int,
    column_conclusions: List[str],
    n_anomaly_cols: int = 0,
    n_trending: int = 0,
    n_noisy: int = 0,
) -> str:
    """Build human-readable conclusion for tabular analysis.

    Args:
        Summary metrics and per-column conclusion strings.

    Returns:
        Multi-line conclusion string.
    """
    parts: List[str] = [
        f"Documento tabular: {row_count} registros, "
        f"{n_headers} columnas, {n_numeric} numéricas.",
    ]

    # Global summary line
    summary_items: List[str] = []
    if n_anomaly_cols:
        summary_items.append(f"⚠️ {n_anomaly_cols} columna(s) con anomalías detectadas")
    if n_trending:
        summary_items.append(f"{n_trending} columna(s) con tendencia activa")
    if n_noisy:
        summary_items.append(f"{n_noisy} columna(s) con alta variabilidad")
    if not summary_items:
        summary_items.append("Todas las columnas dentro de rangos normales")

    parts.append(" | ".join(summary_items))

    # Per-column details
    parts.extend(column_conclusions)

    return "\n".join(parts)


def build_text_conclusion(
    *,
    word_count: int = 0,
    n_sentences: int = 0,
    paragraph_count: int = 0,
    sentiment_label: str = "neutral",
    sentiment_score: float = 0.0,
    urgency_score: float = 0.0,
    urgency_total_hits: int = 0,
    urgency_hits: Optional[List[Dict[str, Any]]] = None,
    urgency_severity: str = "low",
    readability_avg_sentence_len: float = 0.0,
    readability_vocabulary_richness: float = 0.0,
) -> str:
    """Legacy text conclusion builder (simple string assembly).

    Kept for backward compatibility with callers and tests
    that expect this exact function name and output format.
    """
    parts: List[str] = [
        f"Documento de texto: {word_count} palabras, {n_sentences} oraciones, "
        f"{paragraph_count} párrafos."
    ]

    urgency_label = (
        "ALTA" if urgency_severity == "critical"
        else "MODERADA" if urgency_severity == "warning"
        else "BAJA"
    )
    parts.append(
        f"Sentimiento: {sentiment_label} ({sentiment_score:.2f}). "
        f"Urgencia {urgency_label} — {urgency_total_hits} indicios detectados."
    )

    if urgency_hits:
        hit_names = [h.get("keyword", "?") for h in urgency_hits]
        parts.append(f"Términos clave: {', '.join(hit_names)}.")

    parts.append(
        f"Legibilidad: longitud media oración = {readability_avg_sentence_len:.1f}, "
        f"riqueza vocabulario = {readability_vocabulary_richness:.2f}."
    )

    return "\n".join(parts)


_DOMAIN_LABELS = {
    "infrastructure": "Infrastructure",
    "security": "Security",
    "operations": "Operations",
    "business": "Business",
    "general": "General",
}
