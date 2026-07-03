"""Entity extraction from UniversalResult analysis dict.

Extrae entidades, word_count, urgency_score y sentiment_label
del resultado del pipeline universal para consumo de formateadores
de conclusión (conclusion_formatter, result_builder, document_analysis).
"""

from __future__ import annotations

from typing import Any


def extract_entities(analysis_result: Any) -> tuple[list, int]:
    """Extraer lista de entidades y word_count de un resultado de análisis.

    Args:
        analysis_result: UniversalResult (o duck-typed) con .analysis dict.

    Returns:
        (entities_list, word_count) — entidades como strings, conteo de palabras.
    """
    analysis: dict[str, Any] = getattr(analysis_result, "analysis", {})

    # Orden de resolución: analysis dict → metadata → defaults
    entities: list[str] = []
    word_count: int = 0

    if isinstance(analysis, dict):
        entities = analysis.get("entities", [])
        word_count = analysis.get("word_count", 0)

    if not entities and hasattr(analysis_result, "explanation"):
        explanation = analysis_result.explanation
        if hasattr(explanation, "to_dict"):
            exp_dict = explanation.to_dict()
            entities = exp_dict.get("entities", [])

    if not word_count and hasattr(analysis_result, "metadata"):
        meta = analysis_result.metadata
        if isinstance(meta, dict):
            word_count = meta.get("word_count", 0)

    if isinstance(entities, list):
        entities = [str(e) for e in entities if e]

    return entities, word_count


def extract_urgency_sentiment(analysis_result: Any) -> tuple[float, str]:
    """Extraer urgency_score y sentiment_label de un resultado de análisis.

    Args:
        analysis_result: UniversalResult (o duck-typed) con .analysis dict.

    Returns:
        (urgency_score, sentiment_label) — score [0,1] + label categórico.
    """
    analysis: dict[str, Any] = getattr(analysis_result, "analysis", {})

    urgency_score: float = 0.0
    sentiment_label: str = "neutral"

    if isinstance(analysis, dict):
        urgency_score = analysis.get("urgency_score", 0.0)
        sentiment_label = analysis.get("sentiment_label", "neutral")

    # Fallback: perceptions dentro de cognitive
    if not urgency_score and isinstance(analysis, dict):
        cognitive = analysis.get("cognitive", {})
        perceptions = cognitive.get("engine_perceptions", [])
        for p in perceptions:
            if isinstance(p, dict):
                if p.get("engine_name") == "text_urgency":
                    urgency_score = p.get("predicted_value", 0.0)
                elif p.get("engine_name") == "text_sentiment":
                    sentiment_label = p.get("metadata", {}).get("label", "neutral")

    # Fallback: metadata
    if not urgency_score and hasattr(analysis_result, "metadata"):
        meta = analysis_result.metadata
        if isinstance(meta, dict):
            urgency_score = meta.get("urgency_score", 0.0)
            sentiment_label = meta.get("sentiment_label", "neutral")

    # Fallback: try severity if nothing else works
    if not urgency_score and hasattr(analysis_result, "severity"):
        severity = analysis_result.severity
        urgency_score = 0.7 if getattr(severity, "severity", "") == "critical" else 0.0

    return float(urgency_score), str(sentiment_label)
