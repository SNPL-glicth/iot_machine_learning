"""Feature computation service.

Extracted from ml_features.py for modularity.

FIX 2026-02-02: Agregado warm-up y cooldown para evitar falsos positivos.
"""

from __future__ import annotations

import time
import logging
from datetime import datetime, timezone
from statistics import mean, stdev
from typing import Tuple, Optional, Dict

from ..models.ml_features import MLFeatures

logger = logging.getLogger(__name__)

# Constantes de warm-up y cooldown
MIN_READINGS_FOR_FEATURES = 5  # Mínimo de lecturas para calcular features
MIN_WINDOW_AGE_SECONDS = 30.0  # Ventana mínima de 30 segundos
ANOMALY_COOLDOWN_SECONDS = 300.0  # 5 minutos entre anomalías del mismo sensor


class FeatureComputer:
    """Computes ML features from sensor data.
    
    FIX 2026-02-02: Implementa warm-up y cooldown para evitar:
    - Falsos positivos en primeras lecturas (ML-1)
    - Spam de alertas por anomalías repetidas (ML-4)
    """
    
    def __init__(self):
        # Cooldown tracker: sensor_id -> last_anomaly_timestamp
        self._anomaly_cooldowns: Dict[int, float] = {}
    
    def compute_features(
        self,
        window,
        current_value: float,
        current_timestamp: float,
    ) -> Optional[MLFeatures]:
        """Compute features for a sensor reading.
        
        Returns:
            MLFeatures if enough data available, None if in warm-up period.
        """
        # Get values and timestamps from window
        values = window.get_values()
        timestamps = window.get_timestamps()
        
        # =====================================================================
        # VALIDACIÓN 1: WARM-UP - Mínimo de lecturas (ML-1)
        # =====================================================================
        if len(values) < MIN_READINGS_FOR_FEATURES:
            logger.debug(
                "SKIP warm_up: sensor_id=%d readings=%d required=%d",
                window.sensor_id, len(values), MIN_READINGS_FOR_FEATURES
            )
            return None
        
        # =====================================================================
        # VALIDACIÓN 2: WARM-UP - Ventana temporal mínima
        # =====================================================================
        if timestamps:
            window_age = current_timestamp - min(timestamps)
            if window_age < MIN_WINDOW_AGE_SECONDS:
                logger.debug(
                    "SKIP window_age: sensor_id=%d age=%.1fs required=%.1fs",
                    window.sensor_id, window_age, MIN_WINDOW_AGE_SECONDS
                )
                return None
        
        # Basic metrics
        baseline = mean(values)
        baseline_std = stdev(values) if len(values) > 1 else 0.0
        
        # Deviation metrics
        deviation = abs(current_value - baseline)
        deviation_pct = (deviation / baseline * 100) if baseline != 0 else 0.0
        z_score = (current_value - baseline) / baseline_std if baseline_std > 0 else 0.0
        
        # Trend analysis
        trend_slope, trend_direction = self._compute_trend(values, timestamps)
        
        # Stability score (inverse of normalized variance)
        stability_score = self._compute_stability(values, baseline_std)
        
        # Model confidence (based on window size and variance)
        confidence = self._compute_confidence(len(values), baseline_std, baseline)
        
        # Pattern detection
        pattern_detected = self._detect_pattern(trend_slope, stability_score, z_score)
        
        # Anomaly detection with cooldown
        is_anomalous, anomaly_score = self._detect_anomaly_with_cooldown(
            sensor_id=window.sensor_id,
            z_score=z_score,
            deviation_pct=deviation_pct,
            current_timestamp=current_timestamp,
        )
        
        return MLFeatures(
            sensor_id=window.sensor_id,
            timestamp=current_timestamp,
            current_value=current_value,
            baseline=baseline,
            baseline_std=baseline_std,
            deviation=deviation,
            deviation_pct=deviation_pct,
            z_score=z_score,
            trend_slope=trend_slope,
            trend_direction=trend_direction,
            stability_score=stability_score,
            confidence=confidence,
            pattern_detected=pattern_detected,
            is_anomalous=is_anomalous,
            anomaly_score=anomaly_score,
            window_size=len(values),
            model_version="1.0.0",
        )
    
    def _compute_trend(self, values: list[float], timestamps: list[float]) -> Tuple[float, str]:
        """Compute trend slope and direction."""
        if len(values) < 2:
            return 0.0, "stable"
        
        # Simple linear regression
        n = len(values)
        sum_x = sum(timestamps)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(timestamps, values))
        sum_x2 = sum(x * x for x in timestamps)
        
        # Calculate slope
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0, "stable"
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Determine direction
        if abs(slope) < 1e-6:
            direction = "stable"
        elif slope > 0:
            direction = "up"
        else:
            direction = "down"
        
        return slope, direction
    
    def _compute_stability(self, values: list[float], baseline_std: float) -> float:
        """Compute stability score (0-1, higher is more stable)."""
        if len(values) < 2 or baseline_std == 0:
            return 1.0
        
        # Coefficient of variation (normalized)
        mean_val = mean(values)
        if mean_val == 0:
            return 0.0
        
        cv = baseline_std / abs(mean_val)
        
        # Convert to stability score (inverse of CV, normalized)
        stability = 1.0 / (1.0 + cv)
        return min(1.0, max(0.0, stability))
    
    def _compute_confidence(self, window_size: int, baseline_std: float, baseline: float) -> float:
        """Compute model confidence based on data quality."""
        # Base confidence from window size
        size_confidence = min(1.0, window_size / 50.0)  # Full confidence at 50+ samples
        
        # Adjust for variance
        if baseline == 0:
            variance_penalty = 0.5
        else:
            variance_penalty = min(0.5, baseline_std / abs(baseline))
        
        confidence = size_confidence * (1.0 - variance_penalty)
        return max(0.0, min(1.0, confidence))
    
    def _detect_pattern(self, trend_slope: float, stability_score: float, z_score: float) -> str:
        """Detect behavior pattern."""
        if abs(z_score) > 3.0:
            return "ANOMALY"
        elif abs(trend_slope) > 0.1:
            return "TREND_" + ("UP" if trend_slope > 0 else "DOWN")
        elif stability_score < 0.3:
            return "OSCILLATING"
        elif stability_score > 0.8:
            return "STABLE"
        else:
            return "NORMAL"
    
    def _detect_anomaly_with_cooldown(
        self,
        sensor_id: int,
        z_score: float,
        deviation_pct: float,
        current_timestamp: float,
    ) -> Tuple[bool, float]:
        """Detect if current reading is anomalous with cooldown.
        
        FIX 2026-02-02: Implementa cooldown para evitar spam de alertas (ML-4).
        """
        # Anomaly score based on z-score (0-1 normalized)
        anomaly_score = min(1.0, abs(z_score) / 5.0)
        
        # Check if anomaly conditions are met
        would_be_anomalous = abs(z_score) > 2.5 or deviation_pct > 20.0
        
        if not would_be_anomalous:
            return False, anomaly_score
        
        # =====================================================================
        # VALIDACIÓN: COOLDOWN - Evitar spam de anomalías (ML-4)
        # =====================================================================
        last_anomaly = self._anomaly_cooldowns.get(sensor_id, 0.0)
        time_since_last = current_timestamp - last_anomaly
        
        if time_since_last < ANOMALY_COOLDOWN_SECONDS:
            logger.debug(
                "SKIP cooldown: sensor_id=%d remaining=%.1fs",
                sensor_id, ANOMALY_COOLDOWN_SECONDS - time_since_last
            )
            return False, anomaly_score
        
        # Anomaly confirmed - update cooldown
        self._anomaly_cooldowns[sensor_id] = current_timestamp
        logger.info(
            "ANOMALY detected: sensor_id=%d z_score=%.2f deviation_pct=%.2f%%",
            sensor_id, z_score, deviation_pct
        )
        
        return True, anomaly_score
    
    def clear_cooldown(self, sensor_id: int) -> None:
        """Clear cooldown for a sensor (for testing)."""
        self._anomaly_cooldowns.pop(sensor_id, None)
    
    def clear_all_cooldowns(self) -> None:
        """Clear all cooldowns (for testing)."""
        self._anomaly_cooldowns.clear()
