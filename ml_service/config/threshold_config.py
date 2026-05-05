"""Threshold and anomaly classification configuration.

Centralizes:
- Severity score thresholds (AnomalySeverity mapping)
- Adaptive threshold percentiles
- Drift detector parameters
- Text severity axis weights
- Impact detector physical thresholds

Replaces scattered magic numbers across:
- infrastructure/ml/anomaly/adaptive_thresholds.py
- infrastructure/ml/cognitive/drift/error_drift_detector.py
- infrastructure/ml/cognitive/text/impact_detector.py
- domain/entities/results/anomaly.py
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from pydantic import BaseModel, field_validator


class ThresholdConfig(BaseModel):
    """Centralized threshold configuration.

    All numeric thresholds are normalized [0, 1] or standard deviations
    where applicable, making them domain-agnostic.
    """

    # --- Severity Score Thresholds [0, 1] ---
    # Maps normalized anomaly score → AnomalySeverity tier.
    # Default: 0.3 NONE / 0.5 LOW / 0.7 MEDIUM / 0.9 HIGH / ≥0.9 CRITICAL
    ML_SEVERITY_NONE_MAX: float = 0.3
    ML_SEVERITY_LOW_MAX: float = 0.5
    ML_SEVERITY_MEDIUM_MAX: float = 0.7
    ML_SEVERITY_HIGH_MAX: float = 0.9

    # --- Adaptive Threshold Percentiles ---
    # Historical-score percentile used per severity tier.
    # Higher percentile = stricter (needs more extreme history).
    ML_ADAPTIVE_LOW_PERCENTILE: float = 75.0
    ML_ADAPTIVE_MEDIUM_PERCENTILE: float = 85.0
    ML_ADAPTIVE_HIGH_PERCENTILE: float = 95.0
    ML_ADAPTIVE_CRITICAL_PERCENTILE: float = 99.0

    # Adaptive fallback static thresholds (cold-start).
    ML_ADAPTIVE_FALLBACK_LOW: float = 0.5
    ML_ADAPTIVE_FALLBACK_MEDIUM: float = 0.7
    ML_ADAPTIVE_FALLBACK_HIGH: float = 0.85
    ML_ADAPTIVE_FALLBACK_CRITICAL: float = 0.95

    # --- Adaptive Window Settings ---
    ML_ADAPTIVE_WARMUP_SAMPLES: int = 30
    ML_ADAPTIVE_MAX_HISTORY: int = 200

    # --- Drift Detector (Page-Hinkley) ---
    ML_DRIFT_PH_DELTA: float = 0.005
    ML_DRIFT_PH_LAMBDA: float = 50.0
    ML_DRIFT_PH_ALPHA: float = 0.9999

    # --- Drift Detector (ADWIN) ---
    ML_DRIFT_ADWIN_DELTA: float = 0.002
    ML_DRIFT_ADWIN_MAX_WINDOW: int = 1000

    # --- Drift Z-Score Normalization ---
    # Number of rolling std for declaring drift from normalized error.
    ML_DRIFT_ZSCORE_THRESHOLD: float = 3.0  # 3σ standard

    # --- Text Severity 3-Axis Weights ---
    # urgency × w_u + sentiment × w_s + impact × w_i = composite [0, 1]
    ML_TEXT_WEIGHT_URGENCY: float = 0.45
    ML_TEXT_WEIGHT_SENTIMENT: float = 0.20
    ML_TEXT_WEIGHT_IMPACT: float = 0.35

    # --- Text Severity Composite Thresholds ---
    ML_TEXT_THRESHOLD_INFO: float = 0.15
    ML_TEXT_THRESHOLD_WARNING: float = 0.35
    ML_TEXT_THRESHOLD_CRITICAL: float = 0.55

    # --- Urgency Override Thresholds ---
    ML_TEXT_URGENCY_OVERRIDE_CRITICAL: float = 0.85
    ML_TEXT_URGENCY_OVERRIDE_WARNING: float = 0.75

    # --- Impact Detector Physical Thresholds ---
    # Temperatures in °C; CPU/Memory/Disk in %
    ML_IMPACT_TEMP_CRITICAL: float = 80.0
    ML_IMPACT_CPU_CRITICAL: float = 90.0
    ML_IMPACT_MEMORY_CRITICAL: float = 90.0
    ML_IMPACT_DISK_CRITICAL: float = 90.0

    # --- Regime Detector ---
    # Minimum points per regime cluster (statistical derivation).
    ML_REGIME_MIN_POINTS_PER_CLUSTER: int = 10

    # --- Hampel Filter ---
    ML_HAMPEL_K: float = 3.0  # 3σ equivalent via MAD

    @field_validator(
        "ML_SEVERITY_NONE_MAX",
        "ML_SEVERITY_LOW_MAX",
        "ML_SEVERITY_MEDIUM_MAX",
        "ML_SEVERITY_HIGH_MAX",
    )
    @classmethod
    def _validate_monotonic(cls, v: float, info) -> float:
        """Ensure thresholds are strictly monotonic."""
        return max(0.0, min(1.0, v))

    @property
    def score_thresholds(self) -> Tuple[float, float, float, float]:
        """Return (none_max, low_max, medium_max, high_max) tuple."""
        return (
            self.ML_SEVERITY_NONE_MAX,
            self.ML_SEVERITY_LOW_MAX,
            self.ML_SEVERITY_MEDIUM_MAX,
            self.ML_SEVERITY_HIGH_MAX,
        )

    @property
    def adaptive_percentiles(self) -> Dict[str, float]:
        """Return severity → percentile mapping."""
        return {
            "LOW": self.ML_ADAPTIVE_LOW_PERCENTILE,
            "MEDIUM": self.ML_ADAPTIVE_MEDIUM_PERCENTILE,
            "HIGH": self.ML_ADAPTIVE_HIGH_PERCENTILE,
            "CRITICAL": self.ML_ADAPTIVE_CRITICAL_PERCENTILE,
        }

    @property
    def adaptive_fallbacks(self) -> Dict[str, float]:
        """Return cold-start static fallback thresholds."""
        return {
            "LOW": self.ML_ADAPTIVE_FALLBACK_LOW,
            "MEDIUM": self.ML_ADAPTIVE_FALLBACK_MEDIUM,
            "HIGH": self.ML_ADAPTIVE_FALLBACK_HIGH,
            "CRITICAL": self.ML_ADAPTIVE_FALLBACK_CRITICAL,
        }

    @property
    def text_weights(self) -> Tuple[float, float, float]:
        """Return (urgency, sentiment, impact) weights."""
        return (
            self.ML_TEXT_WEIGHT_URGENCY,
            self.ML_TEXT_WEIGHT_SENTIMENT,
            self.ML_TEXT_WEIGHT_IMPACT,
        )

    @property
    def text_thresholds(self) -> Tuple[float, float, float]:
        """Return (info_max, warning_min, critical_min) thresholds."""
        return (
            self.ML_TEXT_THRESHOLD_INFO,
            self.ML_TEXT_THRESHOLD_WARNING,
            self.ML_TEXT_THRESHOLD_CRITICAL,
        )
