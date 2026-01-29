"""ML Features dataclass.

Extracted from ml_features.py for modularity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class MLFeatures:
    """Observable ML features produced for every reading.
    
    These features are ALWAYS produced, not just when anomalies are detected.
    This makes the ML observable and explainable.
    """
    
    sensor_id: int
    timestamp: float  # Unix epoch
    
    # Current value
    current_value: float
    
    # Baseline (expected value from model)
    baseline: float
    baseline_std: float
    
    # Deviation from baseline
    deviation: float           # |current - baseline|
    deviation_pct: float       # deviation / baseline * 100
    z_score: float             # (current - baseline) / std
    
    # Trend analysis
    trend_slope: float         # Rate of change (value/second)
    trend_direction: str       # "up" | "down" | "stable"
    
    # Stability (inverse of normalized variance)
    stability_score: float     # 0.0 (unstable) to 1.0 (very stable)
    
    # Model confidence
    confidence: float          # 0.0 to 1.0
    
    # Pattern classification
    pattern_detected: str      # "STABLE", "TREND_UP", "OSCILLATING", etc.
    
    # Anomaly indicators
    is_anomalous: bool = False
    anomaly_score: float = 0.0
    
    # Metadata
    window_size: int = 0
    model_version: str = "1.0.0"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "sensor_id": self.sensor_id,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(
                self.timestamp, tz=timezone.utc
            ).isoformat(),
            "current_value": round(self.current_value, 4),
            "baseline": round(self.baseline, 4),
            "baseline_std": round(self.baseline_std, 4),
            "deviation": round(self.deviation, 4),
            "deviation_pct": round(self.deviation_pct, 2),
            "z_score": round(self.z_score, 4),
            "trend_slope": round(self.trend_slope, 6),
            "trend_direction": self.trend_direction,
            "stability_score": round(self.stability_score, 4),
            "confidence": round(self.confidence, 4),
            "pattern_detected": self.pattern_detected,
            "is_anomalous": self.is_anomalous,
            "anomaly_score": round(self.anomaly_score, 4),
            "window_size": self.window_size,
            "model_version": self.model_version,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False)
