"""Multivariate anomaly detection subsystem.

Provides PCA-based joint anomaly detection across correlated time series.

Exports:
    MultivariateAnomalyEngine: PCA-based multivariate engine.
    OnlinePCA: Incremental PCA wrapper.
    DynamicCorrelationTracker: Correlation tracking with sliding window.
"""

from .engine import MultivariateAnomalyEngine
from .pca_online import OnlinePCA
from .correlation_tracker import DynamicCorrelationTracker

__all__ = [
    "MultivariateAnomalyEngine",
    "OnlinePCA",
    "DynamicCorrelationTracker",
]
