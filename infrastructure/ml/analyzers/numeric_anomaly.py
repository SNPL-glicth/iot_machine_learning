"""Anomaly detection for numeric columns.

VotingAnomalyDetector ensemble implementation.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Lazy imports — avoid hard failures if numpy/sklearn aren't available
_ml_engines_available = True
try:
    from iot_machine_learning.domain.entities.iot.sensor_reading import SensorReading, SensorWindow
    from iot_machine_learning.infrastructure.ml.anomaly.core.detector import VotingAnomalyDetector
    from iot_machine_learning.infrastructure.ml.anomaly.core.config import AnomalyDetectorConfig
except Exception:
    _ml_engines_available = False


def run_anomaly_detection(
    col_name: str,
    values: List[float],
    timestamps: List[float],
    mean: float,
    std: float,
) -> Dict[str, Any]:
    """Run VotingAnomalyDetector ensemble on column values."""
    if not _ml_engines_available:
        return {"detector": "unavailable", "has_anomalies": False}
    
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
