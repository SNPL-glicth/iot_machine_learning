"""Multivariate anomaly engine — PCA-based joint anomaly detection.

Detects anomalies in correlated time series using incremental PCA.
Registered as 'multivariate_pca' engine.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import numpy as np

from iot_machine_learning.infrastructure.ml.interfaces import (
    PredictionEngine,
    PredictionResult,
)
from iot_machine_learning.infrastructure.ml.engines.core.factory import register_engine

from .pca_online import OnlinePCA
from .correlation_tracker import DynamicCorrelationTracker

logger = logging.getLogger(__name__)


@register_engine("multivariate_pca")
class MultivariateAnomalyEngine(PredictionEngine):
    """Multivariate anomaly detection using incremental PCA.
    
    Detects joint anomalies across correlated time series.
    Requires at least 3 correlated series to activate.
    
    Attributes:
        _pca: Online PCA instance.
        _correlation_tracker: Correlation tracker.
        _min_series: Minimum correlated series required.
        _pca_components: Number of PCA components.
        _threshold_percentile: Percentile for anomaly threshold.
    """
    
    def __init__(
        self,
        min_series: int = 3,
        pca_components: int = 2,
        threshold_percentile: float = 95.0,
        correlation_window: int = 100,
    ) -> None:
        """Initialize multivariate engine.
        
        Args:
            min_series: Minimum correlated series required.
            pca_components: Number of PCA components.
            threshold_percentile: Percentile for Mahalanobis threshold.
            correlation_window: Window size for correlation tracking.
        """
        self._min_series = min_series
        self._pca_components = pca_components
        self._threshold_percentile = threshold_percentile
        
        self._pca = OnlinePCA(n_components=pca_components)
        self._correlation_tracker = DynamicCorrelationTracker(
            window_size=correlation_window,
        )
        self._distance_history: List[float] = []
    
    @property
    def name(self) -> str:
        return "multivariate_pca"
    
    def can_handle(self, n_points: int) -> bool:
        """Can handle if enough points for PCA."""
        return n_points >= 10
    
    def predict(
        self,
        values: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> PredictionResult:
        """Predict using multivariate PCA.
        
        Args:
            values: Time series values.
            timestamps: Optional timestamps.
        
        Returns:
            PredictionResult with anomaly score.
        """
        start_time = time.perf_counter()
        
        # Pass-through if not enough data
        if len(values) < 10:
            return self._passthrough(values, reason="insufficient_data")
        
        # This is a placeholder — actual multivariate detection
        # requires access to correlated series data, which would
        # be injected via context in a real implementation.
        # For now, return pass-through with confidence=0.0
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return PredictionResult(
            predicted_value=values[-1],
            confidence=0.0,  # Low confidence = not used in voting
            trend="stable",
            metadata={
                "engine": "multivariate_pca",
                "latency_ms": latency_ms,
                "passthrough": True,
                "reason": "requires_correlated_series_context",
            },
        )
    
    def _passthrough(
        self,
        values: List[float],
        reason: str = "insufficient_data",
    ) -> PredictionResult:
        """Pass-through prediction."""
        return PredictionResult(
            predicted_value=values[-1] if values else 0.0,
            confidence=0.0,
            trend="stable",
            metadata={
                "engine": "multivariate_pca",
                "passthrough": True,
                "reason": reason,
            },
        )
