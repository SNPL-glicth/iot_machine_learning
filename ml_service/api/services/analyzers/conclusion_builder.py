"""Conclusion rendering using domain Explanation + ExplanationRenderer.

Bridges the gap between raw analysis results and the structured
explainability layer.  Constructs ``Explanation`` value objects and
uses ``ExplanationRenderer`` to produce human-readable output.

Falls back to simple string assembly if the explainability layer
is not available.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from .keyword_config import DOMAIN_KEYWORDS, ACTION_TEMPLATES

logger = logging.getLogger(__name__)

# Lazy imports — explainability layer may not be importable
_explainability_available = True
try:
    from .....domain.entities.explainability.explanation import Explanation, Outcome
    from .....domain.entities.explainability.signal_snapshot import SignalSnapshot
    from .....application.explainability.explanation_renderer import ExplanationRenderer

    _renderer = ExplanationRenderer()
except Exception as exc:
    _explainability_available = False
    logger.warning("[CONCLUSION_BUILDER] Explainability layer not available: %s", exc)


# ── Text conclusions ──────────────────────────────────────────────


def build_text_explanation(
    *,
    document_id: str,
    sentiment_label: str,
    sentiment_score: float,
    urgency_score: float,
    urgency_severity: str,
    urgency_hits: List[Dict[str, Any]],
    readability_avg_sentence_len: float,
    readability_vocabulary_richness: float,
    structural_regime: str = "unknown",
    structural_trend: float = 0.0,
    structural_noise: float = 0.0,
    confidence: float = 0.75,
) -> Optional[Dict[str, Any]]:
    """Build an Explanation value object from text analysis results.

    Returns the explanation dict or None if explainability is unavailable.
    """
    if not _explainability_available:
        return None

    try:
        signal = SignalSnapshot(
            n_points=0,
            mean=urgency_score,
            std=0.0,
            noise_ratio=structural_noise,
            slope=structural_trend,
            regime=structural_regime,
            extra={
                "sentiment": sentiment_label,
                "sentiment_score": sentiment_score,
                "urgency_score": urgency_score,
                "urgency_severity": urgency_severity,
                "vocabulary_richness": readability_vocabulary_richness,
                "avg_sentence_length": readability_avg_sentence_len,
            },
        )

        outcome = Outcome(
            kind="text_analysis",
            confidence=confidence,
            trend=structural_regime,
            is_anomaly=urgency_severity == "critical",
            anomaly_score=urgency_score if urgency_severity == "critical" else None,
            extra={
                "sentiment": sentiment_label,
                "urgency_severity": urgency_severity,
            },
        )

        explanation = Explanation(
            series_id=document_id,
            signal=signal,
            outcome=outcome,
        )

        return explanation.to_dict()
    except Exception as exc:
        logger.warning("[CONCLUSION_BUILDER] Failed to build explanation: %s", exc)
        return None


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
    key_topics = _extract_key_topics(full_text, urgency_hits)

    # 3. Build severity assessment
    severity_line = _build_severity_assessment(
        urgency_severity, urgency_score, sentiment_label, sentiment_score,
    )

    # 4. Build topic summary (WHAT is this about)
    topic_line = _build_topic_summary(domain_label, key_topics, word_count)

    # 5. Build recommended actions (WHAT to do)
    action_line = _get_recommended_actions(domain, urgency_severity, key_topics)

    # 6. Recall context (similar past documents)
    recall_line = _build_recall_context(recall_results)

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


def build_text_conclusion(
    *,
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
) -> str:
    """Build human-readable conclusion for text analysis.

    Legacy format — produces score-based output.  Kept as fallback
    when semantic analysis is not possible.

    Args:
        All text analysis metrics from the sub-modules.

    Returns:
        Multi-line conclusion string.
    """
    parts: List[str] = [
        f"Documento de texto: {word_count} palabras, "
        f"{n_sentences} oraciones, {paragraph_count} párrafos.",

        f"Sentimiento: {sentiment_label} (score: {sentiment_score:.2f}) | "
        f"Urgencia: {urgency_score:.2f}",

        f"Legibilidad: promedio {readability_avg_sentence_len:.0f} palabras/oración | "
        f"Riqueza vocabulario: {readability_vocabulary_richness:.2f}",
    ]

    if structural_available:
        parts.append(
            f"Estructura narrativa: régimen {structural_regime}, "
            f"tendencia={structural_trend:.4f}"
        )

    # Urgency narrative
    if urgency_severity == "critical":
        top_kws = ", ".join(h["keyword"] for h in urgency_hits[:5])
        parts.append(
            f"⚠️ Urgencia ALTA ({urgency_total_hits} indicadores: {top_kws}). "
            f"Se recomienda acción inmediata."
        )
    elif urgency_severity == "warning":
        parts.append(
            f"Urgencia moderada ({urgency_total_hits} indicadores)."
        )
    else:
        parts.append(
            "No se detectaron indicadores de urgencia significativos."
        )

    if embedded_numeric_count > 5:
        parts.append(
            f"Se detectaron {embedded_numeric_count} valores numéricos "
            f"embebidos en el texto."
        )

    return "\n".join(parts)


# ── Semantic conclusion helpers ──────────────────────────────────


_DOMAIN_LABELS = {
    "infrastructure": "Infrastructure",
    "security": "Security",
    "operations": "Operations",
    "business": "Business",
    "general": "General",
}

_SEVERITY_LABELS = {
    "critical": "Critical",
    "warning": "Moderate concern",
    "info": "Informational",
}


def classify_document_domain(text: str) -> str:
    """Classify text into a domain category based on keyword density.

    Returns the domain key with the most keyword matches, or
    ``"general"`` if no clear match.
    """
    text_lower = text.lower()
    scores: Dict[str, int] = {}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[domain] = score

    if not scores:
        return "general"

    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    return best


def _extract_key_topics(
    text: str, urgency_hits: List[Dict[str, Any]]
) -> List[str]:
    """Extract key topics: urgency keywords + numeric values mentioned."""
    topics: List[str] = []

    # Extract identifiers (NODE-xxx, SERVER-xxx, etc.) - PRIORITY
    identifiers = re.findall(
        r'\b([A-Z]{2,}[-_]\d{2,})\b', text,
    )
    # Deduplicate identifiers while preserving order
    seen = set()
    unique_identifiers = []
    for ident in identifiers:
        if ident not in seen:
            seen.add(ident)
            unique_identifiers.append(ident)
    for ident in unique_identifiers[:3]:
        topics.append(ident)

    # Extract notable numeric values with context
    # Pattern: number followed/preceded by a unit or identifier
    numeric_contexts = re.findall(
        r'(\b\d+[.,]?\d*\s*(?:°C|°F|%|GB|MB|TB|ms|sec|hrs?|min|rpm|PSI|bar|V|A|W|kW)\b)',
        text, re.IGNORECASE,
    )
    for nc in numeric_contexts[:3]:
        topics.append(nc.strip())

    # Top urgency keywords (sorted by count) - LOWER PRIORITY
    sorted_hits = sorted(
        urgency_hits, key=lambda h: h.get("count", 0), reverse=True
    )
    for h in sorted_hits[:3]:  # Reduced to make room for entities
        topics.append(h["keyword"])

    # DEBUG: Log what we found
    print(f"[DEBUG] Entity extraction: urgency_hits={len(urgency_hits)}, numeric_contexts={len(numeric_contexts)}, identifiers={len(identifiers)}")
    print(f"[DEBUG] Extracted identifiers: {identifiers}")
    print(f"[DEBUG] Final topics (prioritized): {topics}")

    return topics


def _build_severity_assessment(
    urgency_severity: str,
    urgency_score: float,
    sentiment_label: str,
    sentiment_score: float,
) -> str:
    """Build severity assessment line."""
    severity_text = _SEVERITY_LABELS.get(urgency_severity, "Unknown")

    if urgency_severity == "critical":
        return (
            f"Severity: {severity_text}. "
            f"High urgency detected (score {urgency_score:.2f}) "
            f"with {sentiment_label} sentiment. "
            f"Immediate attention required."
        )
    elif urgency_severity == "warning":
        return (
            f"Severity: {severity_text}. "
            f"Elevated urgency (score {urgency_score:.2f}) "
            f"with {sentiment_label} sentiment. "
            f"Review recommended within 24 hours."
        )
    else:
        return (
            f"Severity: {severity_text}. "
            f"Low urgency (score {urgency_score:.2f}), "
            f"{sentiment_label} sentiment. "
            f"No immediate action required."
        )


def _build_topic_summary(
    domain_label: str,
    key_topics: List[str],
    word_count: int,
) -> str:
    """Build the 'what is this about' line."""
    if key_topics:
        topic_str = ", ".join(key_topics[:5])
        return (
            f"{domain_label} document ({word_count} words). "
            f"Key topics: {topic_str}."
        )
    return f"{domain_label} document ({word_count} words)."


def _get_recommended_actions(
    domain: str, severity: str, key_topics: List[str]
) -> str:
    """Get recommended actions based on domain + severity."""
    templates = ACTION_TEMPLATES.get(domain, ACTION_TEMPLATES["general"])
    base_action = templates.get(severity, templates["info"])

    # Enrich with specific identifiers if found
    identifiers = [t for t in key_topics if re.match(r'^[A-Z]{2,}[-_]\d', t)]
    if identifiers and severity in ("critical", "warning"):
        ids_str = ", ".join(identifiers[:3])
        return f"{base_action} Specifically check: {ids_str}."

    return base_action


def _build_recall_context(
    recall_results: Optional[List[Any]],
) -> str:
    """Build context line from semantic recall results."""
    if not recall_results:
        return ""

    parts: List[str] = ["Similar past documents found:"]
    for r in recall_results[:3]:
        # RecallResult has .filename, .conclusion, .score
        fname = getattr(r, "filename", "") or "unknown"
        conclusion = getattr(r, "conclusion", "") or ""
        score = getattr(r, "score", 0.0)

        if conclusion:
            parts.append(
                f"  - {fname} (similarity {score:.0%}): {conclusion[:200]}"
            )
        else:
            content = getattr(r, "content", "")[:100]
            parts.append(
                f"  - {fname} (similarity {score:.0%}): {content}..."
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
