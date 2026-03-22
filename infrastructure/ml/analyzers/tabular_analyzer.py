"""Full tabular/numeric document analysis pipeline.

Orchestrates per-column ML analysis (structural + anomaly detection)
and assembles the final result dict with triggers, thresholds,
conclusion, and confidence.

Single entry point: ``analyze_tabular_document(payload)``.
"""

from __future__ import annotations

from typing import Any, Dict, List

from iot_machine_learning.infrastructure.ml.cognitive.text.conclusion_builder import build_tabular_conclusion
from iot_machine_learning.infrastructure.ml.analyzers.numeric_analyzer import analyze_numeric_column
from iot_machine_learning.infrastructure.ml.analyzers.numeric_stats import resolve_series


def analyze_tabular_document(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run the full tabular analysis pipeline.

    Args:
        payload: Normalized payload with ``data.raw_series``, etc.

    Returns:
        Result dict with ``analysis``, ``adaptive_thresholds``,
        ``conclusion``, ``confidence``.
    """
    data = payload.get("data", {})
    row_count = data.get("row_count", 0)
    headers = data.get("headers", [])
    raw_series: Dict[str, List[float]] = data.get("raw_series", {})
    numeric_columns = data.get("numeric_columns", [])
    sample_rows = data.get("sample_rows", [])

    analysis: Dict[str, Any] = {
        "structural": {},
        "anomalies": {},
        "patterns": [],
        "triggers_activated": [],
    }
    thresholds: Dict[str, float] = {}
    numeric_stats: List[Dict[str, Any]] = []
    column_conclusions: List[str] = []
    confidence_scores: List[float] = []

    series_data = resolve_series(raw_series, numeric_columns, sample_rows)

    if not series_data:
        return {
            "analysis": analysis,
            "adaptive_thresholds": {},
            "conclusion": (
                f"Documento tabular con {row_count} registros y "
                f"{len(headers)} columnas. Sin datos numéricos para analizar."
            ),
            "confidence": 0.5,
        }

    for col_name, values in series_data.items():
        col_result = analyze_numeric_column(col_name, values)

        numeric_stats.append(col_result.stats)
        analysis["structural"][col_name] = col_result.structural

        if col_result.anomaly_result:
            analysis["anomalies"][col_name] = col_result.anomaly_result

        thresholds.update(col_result.thresholds)
        analysis["triggers_activated"].extend(col_result.triggers)
        column_conclusions.append(col_result.conclusion)
        confidence_scores.append(col_result.confidence)

    n_anomaly_cols = sum(
        1 for a in analysis["anomalies"].values()
        if a.get("has_anomalies")
    )
    n_trending = sum(
        1 for s in analysis["structural"].values()
        if s.get("regime") in ("trending", "accelerating")
    )
    n_noisy = sum(
        1 for s in analysis["structural"].values()
        if s.get("regime") == "noisy"
    )

    conclusion = build_tabular_conclusion(
        row_count=row_count,
        n_headers=len(headers),
        n_numeric=len(series_data),
        column_conclusions=column_conclusions,
        n_anomaly_cols=n_anomaly_cols,
        n_trending=n_trending,
        n_noisy=n_noisy,
    )

    avg_confidence = (
        sum(confidence_scores) / len(confidence_scores)
        if confidence_scores else 0.7
    )
    analysis["numeric_stats"] = numeric_stats

    return {
        "analysis": analysis,
        "adaptive_thresholds": thresholds,
        "conclusion": conclusion,
        "confidence": round(avg_confidence, 3),
    }
