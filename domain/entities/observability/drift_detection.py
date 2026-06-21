"""
DriftDetection domain entity for drift detection results.

This is a domain entity for drift detection results.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass(frozen=True)
class DriftResult:
    """
    Domain entity for drift detection results (DDD value object).
    
    This represents the domain concept of drift detection results.
    """
    
    timestamp: float
    drift_detected: bool
    drift_type: str
    drift_magnitude: float
    drift_sensor_id: Optional[int]
    drift_regime: Optional[str]
    drift_temporal_window: Optional[tuple]
    statistical_drift_score: float
    regime_drift_score: float
    feature_drift_score: float
    anomaly_frequency_drift_score: float
    embedding_drift_score: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "drift_detected": self.drift_detected,
            "drift_type": self.drift_type,
            "drift_magnitude": self.drift_magnitude,
            "drift_sensor_id": self.drift_sensor_id,
            "drift_regime": self.drift_regime,
            "drift_temporal_window": self.drift_temporal_window,
            "statistical_drift_score": self.statistical_drift_score,
            "regime_drift_score": self.regime_drift_score,
            "feature_drift_score": self.feature_drift_score,
            "anomaly_frequency_drift_score": self.anomaly_frequency_drift_score,
            "embedding_drift_score": self.embedding_drift_score,
        }
