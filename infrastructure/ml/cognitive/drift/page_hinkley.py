"""Page-Hinkley drift detector — online concept drift detection.

Detects abrupt changes in the mean of a time series using cumulative sum
of deviations. Lightweight, no external dependencies, thread-safe.

Reference:
    E. S. Page (1954). "Continuous Inspection Schemes".
    Biometrika 41 (1/2): 100–115.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PageHinkleyConfig:
    """Configuration for Page-Hinkley detector.
    
    Attributes:
        delta: Magnitude of changes to detect (smaller = more sensitive).
        lambda_: Detection threshold (larger = fewer false positives).
        alpha: Forgetting factor for mean estimation (0 = no forgetting).
    """
    delta: float
    lambda_: float
    alpha: float


class PageHinkleyDetector:
    """Page-Hinkley test for online drift detection.
    
    Monitors cumulative sum of deviations from running mean.
    Drift is detected when cumsum exceeds lambda threshold.
    
    Thread-safe: no shared mutable state across instances.
    
    Attributes:
        _config: Detector configuration.
        _sum: Cumulative sum of deviations.
        _mean: Running mean estimate.
        _n: Number of observations processed.
    """
    
    def __init__(self, config: PageHinkleyConfig) -> None:
        """Initialize detector.
        
        Args:
            config: Page-Hinkley configuration.
        """
        self._config = config
        self._sum = 0.0
        self._mean = 0.0
        self._n = 0
    
    def update(self, value: float) -> bool:
        """Process new observation and check for drift.
        
        Args:
            value: New observation value.
        
        Returns:
            True if drift detected, False otherwise.
        """
        if self._n == 0:
            self._mean = value
            self._n = 1
            return False
        
        # Update running mean with forgetting factor
        if self._config.alpha > 0:
            self._mean = (
                self._config.alpha * value + 
                (1.0 - self._config.alpha) * self._mean
            )
        else:
            # No forgetting: standard cumulative mean
            self._mean = (self._mean * self._n + value) / (self._n + 1)
        
        # Cumulative sum of deviations
        deviation = value - self._mean - self._config.delta
        self._sum = max(0.0, self._sum + deviation)
        
        self._n += 1
        
        # Drift detected when cumsum exceeds threshold
        drift_detected = self._sum > self._config.lambda_
        
        if drift_detected:
            logger.debug(
                "page_hinkley_drift_detected",
                extra={
                    "cumsum": round(self._sum, 4),
                    "threshold": self._config.lambda_,
                    "mean": round(self._mean, 4),
                    "n_observations": self._n,
                },
            )
        
        return drift_detected
    
    def reset(self) -> None:
        """Reset detector state after drift confirmation."""
        self._sum = 0.0
        self._mean = 0.0
        self._n = 0
    
    @property
    def cumsum(self) -> float:
        """Current cumulative sum value."""
        return self._sum
    
    @property
    def mean(self) -> float:
        """Current running mean estimate."""
        return self._mean
    
    @property
    def n_observations(self) -> int:
        """Number of observations processed."""
        return self._n
