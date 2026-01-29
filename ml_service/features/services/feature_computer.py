"""Feature computation service.

Extracted from ml_features.py for modularity.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from statistics import mean, stdev
from typing import Tuple

from ..models.ml_features import MLFeatures


class FeatureComputer:
    """Computes ML features from sensor data."""
    
    def compute_features(
        self,
        window,
        current_value: float,
        current_timestamp: float,
    ) -> MLFeatures:
        """Compute features for a sensor reading."""
        
        # Get values and timestamps from window
        values = window.get_values()
        timestamps = window.get_timestamps()
        
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
        
        # Anomaly detection
        is_anomalous, anomaly_score = self._detect_anomaly(z_score, deviation_pct)
        
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
    
    def _detect_anomaly(self, z_score: float, deviation_pct: float) -> Tuple[bool, float]:
        """Detect if current reading is anomalous."""
        # Anomaly score based on z-score (0-1 normalized)
        # z_score of 3 = anomaly_score of ~0.5
        # z_score of 5 = anomaly_score of ~0.8
        anomaly_score = min(1.0, abs(z_score) / 5.0)
        
        # Is anomalous if z_score > 2.5 or deviation > 20%
        is_anomalous = abs(z_score) > 2.5 or deviation_pct > 20.0
        
        return is_anomalous, anomaly_score
