"""Legacy conclusion building for text analysis.

Fallback format when semantic analysis is not available.
Produces score-based output.
"""

from __future__ import annotations

from typing import Any, Dict, List


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
