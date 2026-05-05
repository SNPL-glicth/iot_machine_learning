"""Drift detection subsystem — online concept drift detection.

Provides lightweight drift detectors for monitoring time series
concept changes in real-time.

Exports:
    PageHinkleyDetector: Cumulative sum-based drift detection.
    PageHinkleyConfig: Configuration for Page-Hinkley detector.
    ADWINDetector: Adaptive windowing drift detection.
    ErrorDriftDetector: Drift detection based on prediction errors.
"""

from .page_hinkley import PageHinkleyDetector, PageHinkleyConfig
from .adwin import ADWINDetector
from .error_drift_detector import ErrorDriftDetector

__all__ = [
    "PageHinkleyDetector",
    "PageHinkleyConfig",
    "ADWINDetector",
    "ErrorDriftDetector",
]
