"""Baseline tracker — adaptive threshold for multivariate anomaly scores.

Single responsibility: maintain rolling history and compute adaptive baseline.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class BaselineTracker:
    """Tracks baseline threshold for anomaly scores.
    
    Uses rolling window of historical scores to compute adaptive
    threshold based on percentile.
    
    Attributes:
        _baseline_percentile: Percentile for threshold (e.g., 95).
        _warmup_samples: Minimum samples for baseline.
        _max_history: Maximum history size.
        _score_history: Rolling window of scores.
        _baseline_threshold: Current baseline threshold.
    """
    
    def __init__(
        self,
        baseline_percentile: float = 95.0,
        warmup_samples: int = 30,
        max_history: int = 200,
    ) -> None:
        """Initialize baseline tracker.
        
        Args:
            baseline_percentile: Percentile for threshold.
            warmup_samples: Minimum samples for baseline.
            max_history: Maximum history size.
        """
        self._baseline_percentile = baseline_percentile
        self._warmup_samples = warmup_samples
        self._max_history = max_history
        self._score_history: List[float] = []
        self._baseline_threshold: Optional[float] = None
    
    def update(self, score: float) -> None:
        """Update baseline with new score.
        
        Args:
            score: New anomaly score.
        """
        # Add to history
        self._score_history.append(score)
        
        # Maintain max history
        if len(self._score_history) > self._max_history:
            self._score_history.pop(0)
        
        # Update baseline if enough samples
        if len(self._score_history) >= self._warmup_samples:
            self._baseline_threshold = float(np.percentile(
                self._score_history,
                self._baseline_percentile
            ))
    
    def normalize(self, score: float) -> float:
        """Normalize score using baseline threshold.
        
        Args:
            score: Raw anomaly score.
        
        Returns:
            Normalized score [0, 1].
        """
        if self._baseline_threshold is None or self._baseline_threshold == 0:
            # Still in warmup or no variance
            return 0.0
        
        return min(1.0, score / self._baseline_threshold)
    
    @property
    def is_warmed_up(self) -> bool:
        """True if baseline is established."""
        return len(self._score_history) >= self._warmup_samples
    
    @property
    def baseline_threshold(self) -> Optional[float]:
        """Current baseline threshold."""
        return self._baseline_threshold
