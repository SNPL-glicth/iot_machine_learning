"""Feature computation service.

Extracted from ml_features.py for modularity.

FIX 2026-02-02: Agregado warm-up y cooldown para evitar falsos positivos.
"""

from __future__ import annotations

import time
import logging
from datetime import datetime, timezone
from statistics import mean, stdev
from typing import Tuple, Optional, Dict, TYPE_CHECKING

from core.parameters.numerical_constants import STAT_THRESHOLDS
from ..models.ml_features import MLFeatures

if TYPE_CHECKING:
    from infrastructure.ml.cognitive.dynamic.pipeline import DynamicFeaturePipeline
    from infrastructure.ml.cognitive.dynamic.models.feature_config import FeatureConfig

logger = logging.getLogger(__name__)

# Constantes de warm-up y cooldown (fallback retrocompatible)
MIN_READINGS_FOR_FEATURES = 5  # Mínimo de lecturas para calcular features
MIN_WINDOW_AGE_SECONDS = 30.0  # Ventana mínima de 30 segundos
ANOMALY_COOLDOWN_SECONDS = 300.0  # 5 minutos entre anomalías del mismo sensor


class FeatureComputer:
    """Computes ML features from sensor data.

    FIX P3-4: Acepta SensorFeatureConfigRegistry para configuración por sensor.
    FIX 2026-02-02: Implementa warm-up y cooldown.
    FIX 2026-06-19: Agrega DynamicFeaturePipeline para features dinámicas.
    """

    def __init__(self, registry=None, dynamic_pipeline: Optional['DynamicFeaturePipeline'] = None):
        # Cooldown tracker: sensor_id -> last_anomaly_timestamp
        self._anomaly_cooldowns: Dict[int, float] = {}
        self._registry = registry
        self._dynamic_pipeline = dynamic_pipeline

    def compute_features(
        self,
        window,
        current_value: float,
        current_timestamp: float,
        sensor_type: Optional[str] = None,
        dynamic_config: Optional['FeatureConfig'] = None,
    ) -> Optional[MLFeatures]:
        """Compute features. FIX P3-4: config por sensor. FIX 2026-06-19: dynamic features."""
        values = window.get_values(); timestamps = window.get_timestamps()
        config = self._resolve_config(window.sensor_id, sensor_type)
        min_readings = config.min_readings_for_features; min_age = config.min_window_age_seconds
        if len(values) < min_readings:
            logger.debug("SKIP warm_up: sensor_id=%d readings=%d required=%d",
                         window.sensor_id, len(values), min_readings)
            return None
        if timestamps:
            window_age = current_timestamp - min(timestamps)
            if window_age < min_age:
                logger.debug("SKIP window_age: sensor_id=%d age=%.1fs required=%.1fs",
                             window.sensor_id, window_age, min_age)
                return None
        baseline = mean(values); baseline_std = stdev(values) if len(values) > 1 else 0.0
        deviation = abs(current_value - baseline)
        deviation_pct = (deviation / baseline * 100) if baseline != 0 else 0.0
        z_score = (current_value - baseline) / baseline_std if baseline_std > 0 else 0.0
        trend_slope, trend_direction = self._compute_trend(values, timestamps)
        stability_score = self._compute_stability(values, baseline_std)
        confidence = self._compute_confidence(len(values), baseline_std, baseline)
        pattern_detected = self._detect_pattern(trend_slope, stability_score, z_score)
        is_anomalous, anomaly_score = self._detect_anomaly_with_cooldown(
            sensor_id=window.sensor_id, z_score=z_score,
            deviation_pct=deviation_pct, current_timestamp=current_timestamp,
        )
        
        # Compute dynamic features if pipeline is available
        dynamic_features_dict = None
        if self._dynamic_pipeline is not None:
            try:
                dynamic_features = self._dynamic_pipeline.compute(
                    sensor_id=window.sensor_id,
                    sensor_type=sensor_type or "UNKNOWN",
                    values=values,
                    timestamps=timestamps,
                    current_value=current_value,
                    current_timestamp=current_timestamp,
                    config=dynamic_config,
                )
                if dynamic_features and dynamic_features.has_any_features():
                    dynamic_features_dict = dynamic_features.to_dict()
                    # Update model version to indicate dynamic features
                    model_version = "2.0.0"
                else:
                    model_version = "1.0.0"
            except Exception as e:
                logger.warning("Failed to compute dynamic features: sensor_id=%d error=%s",
                             window.sensor_id, str(e))
                model_version = "1.0.0"
        else:
            model_version = "1.0.0"
        
        return MLFeatures(
            sensor_id=window.sensor_id, timestamp=current_timestamp,
            current_value=current_value, baseline=baseline,
            baseline_std=baseline_std, deviation=deviation,
            deviation_pct=deviation_pct, z_score=z_score,
            trend_slope=trend_slope, trend_direction=trend_direction,
            stability_score=stability_score, confidence=confidence,
            pattern_detected=pattern_detected, is_anomalous=is_anomalous,
            anomaly_score=anomaly_score, window_size=len(values),
            model_version=model_version,
            dynamic_features=dynamic_features_dict,
        )
    def _compute_trend(self, values: list[float], timestamps: list[float]) -> Tuple[float, str]:
        if len(values) < 2:
            return 0.0, "stable"
        n = len(values); sum_x = sum(timestamps); sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(timestamps, values)); sum_x2 = sum(x * x for x in timestamps)
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0, "stable"
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        if abs(slope) < 1e-6:
            direction = "stable"
        elif slope > 0:
            direction = "up"
        else:
            direction = "down"
        return slope, direction
    
    def _compute_stability(self, values: list[float], baseline_std: float) -> float:
        if len(values) < 2 or baseline_std == 0:
            return 1.0
        mean_val = mean(values)
        if mean_val == 0:
            return 0.0
        cv = baseline_std / abs(mean_val)
        stability = 1.0 / (1.0 + cv)
        return min(1.0, max(0.0, stability))
    def _compute_confidence(self, window_size: int, baseline_std: float, baseline: float) -> float:
        size_confidence = min(1.0, window_size / 50.0)
        if baseline == 0:
            variance_penalty = 0.5
        else:
            variance_penalty = min(0.5, baseline_std / abs(baseline))
        confidence = size_confidence * (1.0 - variance_penalty)
        return max(0.0, min(1.0, confidence))
    
    def _detect_pattern(self, trend_slope: float, stability_score: float, z_score: float) -> str:
        """Pattern detection."""
        if abs(z_score) > STAT_THRESHOLDS.Z_SCORE_UPPER:
            return "ANOMALY"
        elif abs(trend_slope) > 0.1:
            return "TREND_" + ("UP" if trend_slope > 0 else "DOWN")
        elif stability_score < 0.3:
            return "OSCILLATING"
        elif stability_score > 0.8:
            return "STABLE"
        else:
            return "NORMAL"
    
    def _detect_anomaly_with_cooldown(self, sensor_id: int, z_score: float,
                                       deviation_pct: float, current_timestamp: float) -> Tuple[bool, float]:
        anomaly_score = min(1.0, abs(z_score) / 5.0)
        would_be_anomalous = abs(z_score) > 2.5 or deviation_pct > 20.0
        if not would_be_anomalous:
            return False, anomaly_score
        last_anomaly = self._anomaly_cooldowns.get(sensor_id, 0.0)
        time_since_last = current_timestamp - last_anomaly
        if time_since_last < ANOMALY_COOLDOWN_SECONDS:
            logger.debug("SKIP cooldown: sensor_id=%d remaining=%.1fs",
                         sensor_id, ANOMALY_COOLDOWN_SECONDS - time_since_last)
            return False, anomaly_score
        self._anomaly_cooldowns[sensor_id] = current_timestamp
        logger.info("ANOMALY detected: sensor_id=%d z_score=%.2f deviation_pct=%.2f%%",
                    sensor_id, z_score, deviation_pct)
        return True, anomaly_score
    def clear_cooldown(self, sensor_id: int) -> None:
        self._anomaly_cooldowns.pop(sensor_id, None)
    def clear_all_cooldowns(self) -> None:
        self._anomaly_cooldowns.clear()
    def _resolve_config(self, sensor_id: int, sensor_type: Optional[str]):
        from ..sensor_feature_config import SensorFeatureConfig, DEFAULT_MIN_WINDOW_AGE, DEFAULT_MIN_READINGS
        if self._registry is not None:
            return self._registry.get(sensor_id, sensor_type)
        return SensorFeatureConfig(
            min_window_age_seconds=MIN_WINDOW_AGE_SECONDS,
            min_readings_for_features=MIN_READINGS_FOR_FEATURES,
        )
