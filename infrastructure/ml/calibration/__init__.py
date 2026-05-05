"""Calibration subsystem — confidence and probability calibration.

Provides temperature scaling and regime-aware calibration for converting
raw anomaly scores into calibrated probabilities.
"""

from .confidence_calibrator import ConfidenceCalibrator

__all__ = [
    "ConfidenceCalibrator",
]
