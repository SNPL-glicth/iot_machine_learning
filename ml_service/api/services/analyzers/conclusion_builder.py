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
