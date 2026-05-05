"""Numeric perception collection logic."""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

from ...analysis.types import EnginePerception
from iot_machine_learning.infrastructure.ml.patterns.change_point_detector import CUSUMDetector
from iot_machine_learning.infrastructure.ml.patterns.regime_detector import RegimeDetector
from iot_machine_learning.infrastructure.ml.anomaly.core.detector import VotingAnomalyDetector
from iot_machine_learning.infrastructure.ml.anomaly.core.config import AnomalyDetectorConfig
from iot_machine_learning.domain.validators.structural_analysis import compute_structural_analysis
from iot_machine_learning.domain.entities.iot.sensor_reading import SensorReading, SensorWindow

logger = logging.getLogger(__name__)


def collect_numeric_perceptions(
    values: list,
    timestamps: Optional[list],
    metadata: Dict[str, Any],
) -> List[EnginePerception]:
    """Build perceptions for numeric series.

    REUSES existing infrastructure/ml/ components:
        - numeric_drift: CUSUMDetector
        - numeric_regime: RegimeDetector
        - numeric_anomaly: VotingAnomalyDetector
        - numeric_structural: compute_structural_analysis()
    """
    perceptions: List[EnginePerception] = []
    n = len(values)

    if n < 3:
        return perceptions

    if timestamps is None:
        timestamps = [float(i) for i in range(n)]

    try:
        structural = compute_structural_analysis(values, timestamps)
        perceptions.append(EnginePerception(
            engine_name="numeric_structural",
            predicted_value=round(structural.stability, 4),
            confidence=0.8,
            trend="up" if structural.slope > 0.1 else "down" if structural.slope < -0.1 else "stable",
            stability=round(structural.stability, 4),
            local_fit_error=round(structural.noise_ratio, 4),
            metadata={
                "regime": structural.regime.value if hasattr(structural.regime, 'value') else str(structural.regime),
                "slope": structural.slope,
                "curvature": structural.curvature,
            },
        ))
    except Exception as e:
        logger.debug(f"structural_analysis failed: {e}")

    perceptions.append(EnginePerception(
        engine_name="numeric_spikes",
        predicted_value=0.0,
        confidence=0.0,
        trend="stable",
        stability=1.0,
        local_fit_error=0.0,
        metadata={"reason": "delta_spike_classifier_removed"},
    ))

    if n >= 20:
        try:
            detector = CUSUMDetector()
            changes = detector.detect(values)
            n_changes = len(changes) if changes else 0
            drift_score = min(1.0, n_changes / 5.0)

            perceptions.append(EnginePerception(
                engine_name="numeric_drift",
                predicted_value=round(1.0 - drift_score, 4),
                confidence=0.75,
                trend="up" if n_changes > 0 else "stable",
                stability=round(1.0 - drift_score, 4),
                local_fit_error=round(drift_score, 4),
                metadata={"n_change_points": n_changes},
            ))
        except Exception as e:
            logger.debug(f"cusum failed: {e}")

        try:
            regime_det = RegimeDetector()
            regime_result = regime_det.detect(values)
            n_regimes = len(set(regime_result.labels)) if hasattr(regime_result, 'labels') else 1
            regime_consistency = 1.0 / n_regimes

            perceptions.append(EnginePerception(
                engine_name="numeric_regime",
                predicted_value=round(regime_consistency, 4),
                confidence=0.7,
                trend="stable",
                stability=round(regime_consistency, 4),
                local_fit_error=round(1.0 - regime_consistency, 4),
                metadata={"n_regimes": n_regimes},
            ))
        except Exception as e:
            logger.debug(f"regime_detector failed: {e}")

        try:
            config = AnomalyDetectorConfig(min_training_points=20)
            anom_detector = VotingAnomalyDetector(config=config)
            anom_detector.train(values, timestamps=timestamps)

            readings = [
                SensorReading(sensor_id=0, value=v, timestamp=t)
                for v, t in zip(values, timestamps)
            ]
            window = SensorWindow(sensor_id=0, readings=readings)
            result = anom_detector.detect(window)

            perceptions.append(EnginePerception(
                engine_name="numeric_anomaly",
                predicted_value=round(1.0 - result.score, 4),
                confidence=round(result.confidence, 4),
                trend="up" if result.is_anomaly else "stable",
                stability=round(1.0 - result.score, 4),
                local_fit_error=round(result.score, 4),
                metadata={
                    "is_anomaly": result.is_anomaly,
                    "severity": result.severity.value,
                },
            ))
        except Exception as e:
            logger.debug(f"voting_anomaly failed: {e}")

    return perceptions


def collect_tabular_perceptions(
    data: dict,
    metadata: Dict[str, Any],
) -> List[EnginePerception]:
    """Build perceptions for tabular data."""
    numeric_columns = metadata.get("numeric_columns", [])
    if not numeric_columns:
        return []
    first_col = numeric_columns[0]
    values = data.get(first_col, [])
    return collect_numeric_perceptions(values, None, metadata)
