"""Adaptive Thresholds — per-series percentile-based anomaly thresholds.

Replaces static thresholds (0.3, 0.5, 0.8) with adaptive percentiles
computed from historical anomaly scores.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Dict, Optional

import numpy as np

from iot_machine_learning.domain.entities.results.anomaly import AnomalySeverity

logger = logging.getLogger(__name__)


class AdaptiveThresholdManager:
    """Manages per-series adaptive anomaly thresholds.
    
    Uses percentile-based thresholds computed from historical scores.
    Provides fallback to static thresholds during cold start.
    
    Attributes:
        warmup_samples: Minimum samples before adaptive thresholds activate
        max_history: Maximum history size per series (memory bound)
    """
    
    def __init__(
        self,
        warmup_samples: Optional[int] = None,
        max_history: Optional[int] = None,
        percentiles: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize adaptive threshold manager.

        Args:
            warmup_samples: Minimum samples for adaptive mode (default from config).
            max_history: Maximum history per series (default from config).
            percentiles: Severity percentiles. Defaults from ThresholdConfig.
        """
        # Lazy-load defaults from centralized config to avoid cross-layer imports at import time.
        try:
            from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
            defaults = FeatureFlags()
        except Exception:
            defaults = None

        self._warmup = (
            warmup_samples
            if warmup_samples is not None
            else getattr(defaults, "ML_ADAPTIVE_WARMUP_SAMPLES", 30)
        )
        self._max_history = (
            max_history
            if max_history is not None
            else getattr(defaults, "ML_ADAPTIVE_MAX_HISTORY", 200)
        )

        if percentiles is not None:
            self._percentiles = percentiles
        elif defaults is not None:
            self._percentiles = defaults.adaptive_percentiles
        else:
            self._percentiles = {
                "LOW": 75.0,
                "MEDIUM": 85.0,
                "HIGH": 95.0,
                "CRITICAL": 99.0,
            }
        
        # Per-series score history (bounded deque)
        self._history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._max_history)
        )
    
    def update(self, series_id: str, score: float) -> None:
        """Record anomaly score for series.
        
        Args:
            series_id: Series identifier
            score: Anomaly score ∈ [0, 1]
        """
        if not 0.0 <= score <= 1.0:
            logger.warning(
                "adaptive_threshold_invalid_score",
                extra={
                    "series_id": series_id,
                    "score": score,
                    "action": "clamped_to_[0,1]",
                },
            )
            score = max(0.0, min(1.0, score))
        
        self._history[series_id].append(score)
    
    def get_threshold(
        self,
        series_id: str,
        severity: str,
        fallback: Optional[float] = None,
    ) -> float:
        """Get adaptive threshold for series and severity.
        
        Args:
            series_id: Series identifier
            severity: Severity level (LOW, MEDIUM, HIGH, CRITICAL)
            fallback: Static fallback if not warmed up
        
        Returns:
            Adaptive threshold or fallback
        """
        history = self._history.get(series_id, deque())
        
        # Cold start: use fallback
        if len(history) < self._warmup:
            if fallback is not None:
                return fallback
            # Default static fallbacks from centralized config
            try:
                from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
                defaults = FeatureFlags()
                static = defaults.adaptive_fallbacks
            except Exception:
                static = {
                    "LOW": 0.5,
                    "MEDIUM": 0.7,
                    "HIGH": 0.85,
                    "CRITICAL": 0.95,
                }
            return static.get(severity, 0.8)
        
        # Adaptive: compute percentile
        percentile = self._percentiles.get(severity, 90.0)
        threshold = float(np.percentile(list(history), percentile))
        
        return threshold
    
    def classify_severity(
        self,
        series_id: str,
        score: float,
    ) -> AnomalySeverity:
        """Classify severity using adaptive thresholds.

        Delegates boundary mapping to ``ThresholdPolicy`` for unified logic.

        Args:
            series_id: Series identifier
            score: Anomaly score

        Returns:
            AnomalySeverity enum
        """
        # Update history
        self.update(series_id, score)

        # Get adaptive thresholds (fallbacks mirror default policy)
        critical_th = self.get_threshold(series_id, "CRITICAL", fallback=0.95)
        high_th = self.get_threshold(series_id, "HIGH", fallback=0.85)
        medium_th = self.get_threshold(series_id, "MEDIUM", fallback=0.7)
        low_th = self.get_threshold(series_id, "LOW", fallback=0.5)

        # Use ThresholdPolicy for unified boundary logic
        from iot_machine_learning.domain.policies.threshold_policy import ThresholdPolicy
        policy = ThresholdPolicy(
            score_thresholds=(low_th, medium_th, high_th, critical_th)
        )
        return policy.classify_score(score)
    
    def is_warmed_up(self, series_id: str) -> bool:
        """Check if series has enough history for adaptive thresholds.
        
        Args:
            series_id: Series identifier
        
        Returns:
            True if warmed up
        """
        return len(self._history.get(series_id, deque())) >= self._warmup
    
    def get_metrics(self, series_id: str) -> Dict[str, object]:
        """Get diagnostic metrics for series.
        
        Args:
            series_id: Series identifier
        
        Returns:
            Metrics dict
        """
        history = self._history.get(series_id, deque())
        
        if len(history) == 0:
            return {
                "n_samples": 0,
                "warmed_up": False,
            }
        
        scores = list(history)
        return {
            "n_samples": len(scores),
            "warmed_up": len(scores) >= self._warmup,
            "mean_score": float(np.mean(scores)),
            "p50": float(np.percentile(scores, 50)),
            "p95": float(np.percentile(scores, 95)),
            "p99": float(np.percentile(scores, 99)),
        }
