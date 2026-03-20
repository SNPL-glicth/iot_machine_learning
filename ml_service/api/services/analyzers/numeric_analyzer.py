"""Numeric column analysis using real ML engines.

Handles per-column structural analysis, anomaly detection, statistical
triggers, and adaptive thresholds.  Uses:
- ``compute_structural_analysis`` for slope, curvature, regime, stability
- ``VotingAnomalyDetector`` ensemble (8 sub-detectors)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MIN_POINTS_FOR_STRUCTURAL = 3
_MIN_POINTS_FOR_ANOMALY = 20

# Lazy imports — avoid hard failures if numpy/sklearn aren't available
_ml_engines_available = True
try:
    from .....domain.validators.structural_analysis import compute_structural_analysis
    from .....domain.entities.structural_analysis import StructuralAnalysis
    from .....domain.entities.iot.sensor_reading import SensorReading, SensorWindow
    from .....infrastructure.ml.anomaly.core.detector import VotingAnomalyDetector
    from .....infrastructure.ml.anomaly.core.config import AnomalyDetectorConfig
except Exception as exc:
    _ml_engines_available = False
    logger.warning("[NUMERIC_ANALYZER] ML engines not available: %s", exc)


@dataclass(frozen=True)
class NumericColumnResult:
    """Result of analyzing a single numeric column.

    Attributes:
        stats: Basic statistics dict.
        structural: Structural analysis dict.
        anomaly_result: Anomaly detection result dict (or None).
        thresholds: Adaptive thresholds dict.
        triggers: List of trigger dicts.
        conclusion: Per-column conclusion string.
        confidence: Confidence score [0, 1].
    """

    stats: Dict[str, Any]
    structural: Dict[str, Any]
    anomaly_result: Optional[Dict[str, Any]]
    thresholds: Dict[str, float]
    triggers: List[Dict[str, Any]]
    conclusion: str
    confidence: float


def analyze_numeric_column(
    col_name: str,
    values: List[float],
) -> NumericColumnResult:
    """Run full ML pipeline on a single numeric column.

    Args:
        col_name: Column name.
        values: List of finite float values.

    Returns:
        ``NumericColumnResult`` with all analysis products.
    """
    n = len(values)
    timestamps = [float(i) for i in range(n)]

    # ── Basic stats (always available) ──
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
    std = math.sqrt(variance)
    v_min, v_max = min(values), max(values)

    stats: Dict[str, Any] = {
        "column": col_name,
        "n_points": n,
        "mean": round(mean, 6),
        "std": round(std, 6),
        "min": round(v_min, 6),
        "max": round(v_max, 6),
    }

    # ── Adaptive thresholds (statistical) ──
    thresholds = {
        f"{col_name}_warning": round(mean + 1.5 * std, 6),
        f"{col_name}_critical": round(mean + 2.5 * std, 6),
        f"{col_name}_min": round(mean - 2.0 * std, 6),
    }

    triggers: List[Dict[str, Any]] = []
    structural_dict: Dict[str, Any] = {}
    anomaly_result: Optional[Dict[str, Any]] = None
    conclusion_lines: List[str] = []
    confidence = 0.7

    # ── Structural Analysis (needs ≥3 points) ──
    if _ml_engines_available and n >= _MIN_POINTS_FOR_STRUCTURAL:
        try:
            sa: StructuralAnalysis = compute_structural_analysis(
                values, timestamps
            )
            structural_dict = sa.to_dict()
            stats["regime"] = sa.regime.value
            stats["trend_strength"] = round(sa.trend_strength, 6)
            stats["stability"] = round(sa.stability, 6)
            stats["noise_ratio"] = round(sa.noise_ratio, 6)

            conclusion_lines.append(
                f"[{col_name}] Régimen: {sa.regime.value} | "
                f"Tendencia: {sa.trend_strength:.4f} | "
                f"Estabilidad: {sa.stability:.4f} | "
                f"Ruido: {sa.noise_ratio:.4f}"
            )

            if sa.is_trending:
                direction = "ascendente" if sa.slope > 0 else "descendente"
                triggers.append({
                    "type": "warning",
                    "field": col_name,
                    "value": sa.trend_strength,
                    "threshold": 0.1,
                    "message": f"{col_name}: tendencia {direction} "
                               f"(pendiente={sa.slope:.6f})",
                })

            confidence = min(0.95, 0.7 + 0.05 * min(n, 5))
        except Exception as exc:
            logger.warning(
                "[NUMERIC_ANALYZER] Structural analysis failed for %s: %s",
                col_name, exc,
            )
    else:
        conclusion_lines.append(
            f"[{col_name}] μ={mean:.4f}, σ={std:.4f}, "
            f"rango=[{v_min:.4f}, {v_max:.4f}]"
        )

    # ── Anomaly Detection (needs ≥20 points) ──
    if _ml_engines_available and n >= _MIN_POINTS_FOR_ANOMALY:
        try:
            anomaly_result = _run_anomaly_detection(
                col_name, values, timestamps, mean, std
            )
            if anomaly_result and anomaly_result.get("has_anomalies"):
                n_anom = anomaly_result["n_anomalies"]
                max_score = anomaly_result["max_score"]
                conclusion_lines.append(
                    f"  Anomalías: {n_anom} detectadas "
                    f"(score máx: {max_score:.3f}, "
                    f"método: voting ensemble 8 detectores)"
                )
                triggers.append({
                    "type": "critical" if max_score > 0.7 else "warning",
                    "field": col_name,
                    "value": max_score,
                    "threshold": 0.5,
                    "message": f"{col_name}: {n_anom} anomalía(s) "
                               f"detectadas por ensemble ML",
                })
                confidence = min(0.95, confidence + 0.05)
            else:
                conclusion_lines.append(
                    "  Anomalías: ninguna (ensemble 8 detectores)"
                )
        except Exception as exc:
            logger.warning(
                "[NUMERIC_ANALYZER] Anomaly detection failed for %s: %s",
                col_name, exc,
            )

    # ── Statistical triggers (always) ──
    for i, v in enumerate(values):
        if v > mean + 2.5 * std:
            triggers.append({
                "type": "critical",
                "field": col_name,
                "value": v,
                "threshold": mean + 2.5 * std,
                "message": f"{col_name}[{i}]={v:.4f} superó umbral "
                           f"crítico (μ+2.5σ={mean + 2.5 * std:.4f})",
            })
        elif v > mean + 1.5 * std:
            triggers.append({
                "type": "warning",
                "field": col_name,
                "value": v,
                "threshold": mean + 1.5 * std,
                "message": f"{col_name}[{i}]={v:.4f} superó umbral "
                           f"advertencia (μ+1.5σ={mean + 1.5 * std:.4f})",
            })

    return NumericColumnResult(
        stats=stats,
        structural=structural_dict,
        anomaly_result=anomaly_result,
        thresholds=thresholds,
        triggers=triggers,
        conclusion="\n".join(conclusion_lines),
        confidence=confidence,
    )


def _run_anomaly_detection(
    col_name: str,
    values: List[float],
    timestamps: List[float],
    mean: float,
    std: float,
) -> Dict[str, Any]:
    """Run VotingAnomalyDetector ensemble on column values."""
    config = AnomalyDetectorConfig(
        min_training_points=min(20, len(values)),
        voting_threshold=0.5,
    )
    detector = VotingAnomalyDetector(config=config)

    # Train on the column's own data
    detector.train(values, timestamps=timestamps)

    # Pre-filter: only scan points beyond 1σ from mean (statistical
    # candidates) to avoid O(n) full ensemble evaluations.
    # Cap at 30 scans max for performance.
    _MAX_SCANS = 30
    candidates = [
        i for i in range(len(values))
        if abs(values[i] - mean) > std
    ]
    # Prioritize by distance from mean (most suspicious first)
    candidates.sort(key=lambda i: abs(values[i] - mean), reverse=True)
    candidates = candidates[:_MAX_SCANS]

    anomalies_found: List[Dict[str, Any]] = []

    for i in candidates:
        # Build a SensorWindow ending at index i
        window_start = max(0, i - 19)
        readings = [
            SensorReading(
                sensor_id=0,
                value=values[j],
                timestamp=timestamps[j],
            )
            for j in range(window_start, i + 1)
        ]
        window = SensorWindow(sensor_id=0, readings=readings)

        result = detector.detect(window)
        if result.is_anomaly:
            anomalies_found.append({
                "index": i,
                "value": values[i],
                "score": round(result.score, 4),
                "severity": result.severity.value,
                "votes": {
                    k: round(v, 3) for k, v in result.method_votes.items()
                },
                "explanation": result.explanation,
            })

    # Sort by index for consistent output
    anomalies_found.sort(key=lambda a: a["index"])

    return {
        "detector": "voting_ensemble_8",
        "has_anomalies": len(anomalies_found) > 0,
        "n_anomalies": len(anomalies_found),
        "max_score": max(
            (a["score"] for a in anomalies_found), default=0.0
        ),
        "anomalies": anomalies_found[:10],
    }


def resolve_series(
    raw_series: Dict[str, List[float]],
    numeric_columns: List[str],
    sample_rows: List[Dict],
) -> Dict[str, List[float]]:
    """Resolve numeric column values from raw_series or sample_rows.

    Args:
        raw_series: Direct column→values mapping from parser.
        numeric_columns: Names of numeric columns.
        sample_rows: Fallback: list of row dicts.

    Returns:
        Dict of column_name → list of finite float values.
    """
    result: Dict[str, List[float]] = {}

    # Prefer raw_series (full data from .NET parser)
    if raw_series:
        for col_name, vals in raw_series.items():
            clean = [float(v) for v in vals if _is_finite(v)]
            if clean:
                result[col_name] = clean
        return result

    # Fallback: extract from sample_rows
    for col in numeric_columns:
        vals = []
        for row in sample_rows:
            raw = row.get(col)
            if raw is not None:
                try:
                    v = float(raw)
                    if math.isfinite(v):
                        vals.append(v)
                except (ValueError, TypeError):
                    continue
        if vals:
            result[col] = vals

    return result


def _is_finite(v) -> bool:
    """Check if a value is a finite number."""
    try:
        return math.isfinite(float(v))
    except (ValueError, TypeError):
        return False
