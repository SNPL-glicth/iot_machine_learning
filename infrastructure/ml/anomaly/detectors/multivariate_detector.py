"""Multivariate sub-detector — joint anomaly detection across correlated series.

Single responsibility: detect joint anomalies using PCA on correlated series.
Implements SubDetector protocol for VotingAnomalyDetector integration.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Dict

import numpy as np

from ..core.protocol import SubDetector
from iot_machine_learning.infrastructure.ml.engines.multivariate import (
    OnlinePCA,
    DynamicCorrelationTracker,
)
from .multivariate import MatrixBuilder, BaselineTracker

logger = logging.getLogger(__name__)


class MultivariateDetector(SubDetector):
    """Sub-detector for multivariate anomalies.
    
    Detects joint anomalies across correlated time series using PCA.
    Pass-through (vote=0.0) when insufficient correlated series.
    
    Uses adaptive baseline and dynamic threshold based on historical scores.
    
    Attributes:
        _min_series: Minimum correlated series required.
        _enabled: Whether multivariate detection is enabled.
        _pca_components: Number of PCA components.
        _baseline_percentile: Percentile for baseline threshold (e.g., 95).
        _warmup_samples: Minimum samples for baseline establishment.
        _pca: Online PCA instance (lazy-initialized).
        _score_history: Historical scores for adaptive threshold.
        _baseline_threshold: Adaptive threshold (95th percentile of history).
    """
    
    def __init__(
        self,
        min_series: int = 3,
        enabled: bool = False,
        pca_components: int = 2,
        baseline_percentile: float = 95.0,
        warmup_samples: int = 30,
        max_history: int = 200,
    ) -> None:
        """Initialize multivariate detector.
        
        Args:
            min_series: Minimum correlated series required.
            enabled: Whether multivariate detection is enabled.
            pca_components: Number of PCA components.
            baseline_percentile: Percentile for adaptive threshold (95 = 95th percentile).
            warmup_samples: Minimum samples before baseline is established.
            max_history: Maximum score history size.
        """
        self._min_series = min_series
        self._enabled = enabled
        self._pca_components = pca_components
        self._pca: Optional[OnlinePCA] = None
        self._baseline_tracker = BaselineTracker(
            baseline_percentile=baseline_percentile,
            warmup_samples=warmup_samples,
            max_history=max_history,
        )
        self._correlation_tracker = DynamicCorrelationTracker(
            window_size=100,
            min_samples=warmup_samples,
        )
    
    @property
    def method_name(self) -> str:
        return "multivariate"
    
    def train(self, values: List[float], **kwargs: object) -> None:
        """Train multivariate detector.
        
        Currently no-op as multivariate detection uses online PCA.
        """
        pass
    
    def vote(self, value: float, **kwargs: object) -> Optional[float]:
        """Produce vote for a single value.
        
        Args:
            value: Value to evaluate.
            **kwargs: Additional context.
        
        Returns:
            Vote [0.0, 1.0] or None.
        """
        # Multivariate detection requires full series context
        # Single value voting not supported
        return None
    
    @property
    def is_trained(self) -> bool:
        """True if detector is trained."""
        # Multivariate uses online learning, always "trained"
        return True
    
    def detect(
        self,
        values: List[float],
        timestamps: Optional[List[float]] = None,
        **kwargs: object,
    ) -> float:
        """Detect multivariate anomaly.
        
        Args:
            values: Time series values.
            timestamps: Optional timestamps.
            **kwargs: Additional context (correlated_series_data: Dict[str, List[float]]).
        
        Returns:
            Anomaly vote [0.0, 1.0]. Returns 0.0 if disabled or
            insufficient correlated series.
        """
        # Early exit if disabled
        if not self._enabled:
            return 0.0
        
        # Early exit if insufficient data
        if len(values) < 10:
            return 0.0
        
        # Get correlated series data from kwargs
        correlated_series_data: Dict[str, List[float]] = kwargs.get('correlated_series_data', {})
        
        if len(correlated_series_data) < self._min_series - 1:
            # Not enough correlated series (target series + correlated)
            logger.debug(
                "multivariate_detector_insufficient_series",
                extra={
                    "n_correlated": len(correlated_series_data),
                    "min_required": self._min_series - 1,
                },
            )
            return 0.0
        
        # Get series_id from kwargs
        series_id_param = kwargs.get('series_id', None)
        
        # Filter by correlation if series_id provided
        if series_id_param:
            correlated_series_data = self._filter_by_correlation(
                series_id_param,
                values,
                correlated_series_data,
            )
        
        # Build multivariate matrix X
        try:
            X = MatrixBuilder.build(
                target_values=values,
                correlated_series_data=correlated_series_data,
                min_samples=self._pca_components,
            )
            
            if X is None or X.shape[0] < self._pca_components:
                return 0.0
            
            # Lazy-initialize PCA
            if self._pca is None:
                self._pca = OnlinePCA(n_components=min(self._pca_components, X.shape[1]))
            
            # Incremental fit
            self._pca.partial_fit(X)
            
            # Score samples
            scores = self._pca.score_samples(X)
            
            if scores is None or len(scores) == 0:
                return 0.0
            
            # Use latest score (most recent observation)
            anomaly_score = float(scores[-1])
            
            # Update baseline tracker
            self._baseline_tracker.update(anomaly_score)
            
            # Normalize score
            score_norm = self._baseline_tracker.normalize(anomaly_score)
            
            logger.debug(
                "multivariate_detector_score",
                extra={
                    "raw_score": round(anomaly_score, 4),
                    "baseline_threshold": round(self._baseline_tracker.baseline_threshold, 4) if self._baseline_tracker.baseline_threshold else None,
                    "normalized_score": round(score_norm, 4),
                    "n_series": X.shape[1],
                    "n_samples": X.shape[0],
                    "warmup_complete": self._baseline_tracker.is_warmed_up,
                },
            )
            
            return score_norm
        
        except Exception as e:
            logger.error(
                "multivariate_detector_error",
                extra={
                    "event": "DETECTOR_ERROR",
                    "error": str(e),
                    "action_taken": "return_zero",
                },
            )
            return 0.0
    
    def _filter_by_correlation(
        self,
        series_id: str,
        target_values: List[float],
        correlated_series_data: Dict[str, List[float]],
    ) -> Dict[str, List[float]]:
        """Filter correlated series by correlation threshold.
        
        Uses DynamicCorrelationTracker to select highly correlated series.
        
        Args:
            series_id: Target series ID.
            target_values: Target series values.
            correlated_series_data: Dict of {series_id: values}.
        
        Returns:
            Filtered dict with only correlated series.
        """
        try:
            # Determine window size
            window_size = len(target_values)
            for series_values in correlated_series_data.values():
                window_size = min(window_size, len(series_values))
            
            # Update correlation tracker
            for val in target_values[-window_size:]:
                self._correlation_tracker.update(series_id, val)
            
            for sid, svalues in correlated_series_data.items():
                for val in svalues[-window_size:]:
                    self._correlation_tracker.update(sid, val)
            
            # Get correlated series (threshold=0.5)
            correlated = self._correlation_tracker.get_correlated(
                series_id,
                threshold=0.5,
            )
            
            # Filter
            if correlated:
                correlated_ids = {sid for sid, _ in correlated}
                filtered = {
                    sid: vals
                    for sid, vals in correlated_series_data.items()
                    if sid in correlated_ids
                }
                
                logger.debug(
                    "multivariate_correlation_filter",
                    extra={
                        "series_id": series_id,
                        "n_candidates": len(correlated),
                        "n_selected": len(filtered),
                        "top_correlations": [
                            {"series": sid, "corr": round(corr, 3)}
                            for sid, corr in correlated[:3]
                        ],
                    },
                )
                
                return filtered
            
            return correlated_series_data
        
        except Exception as e:
            logger.warning(
                "correlation_filter_failed",
                extra={
                    "event": "FILTER_ERROR",
                    "error": str(e),
                    "action_taken": "use_all_series",
                },
            )
            return correlated_series_data
