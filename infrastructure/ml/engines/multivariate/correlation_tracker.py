"""Dynamic correlation tracker — maintains sliding window correlations.

Tracks correlations between time series using sliding window.
IEC 62443 compliant: validates series_id before use as key.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from typing import Dict, List, Optional, Tuple, Deque

import numpy as np

logger = logging.getLogger(__name__)

# IEC 62443: series_id validation pattern
_SERIES_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-:]+$')


class DynamicCorrelationTracker:
    """Tracks correlations between time series.
    
    Maintains sliding window of recent observations per series
    and computes pairwise Pearson correlations.
    
    IEC 62443: Validates series_id to prevent injection attacks.
    
    Attributes:
        _window_size: Maximum window size per series.
        _min_samples: Minimum samples required for correlation.
        _data: Sliding windows per series.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        min_samples: int = 10,
    ) -> None:
        """Initialize correlation tracker.
        
        Args:
            window_size: Maximum window size.
            min_samples: Minimum samples for correlation.
        
        Raises:
            ValueError: If window_size < min_samples.
        """
        if window_size < min_samples:
            raise ValueError(f"window_size must be >= min_samples")
        
        self._window_size = window_size
        self._min_samples = min_samples
        self._data: Dict[str, Deque[float]] = {}
    
    def _validate_series_id(self, series_id: str) -> bool:
        """Validate series_id against IEC 62443 pattern.
        
        Prevents path traversal and key injection attacks.
        """
        if not _SERIES_ID_PATTERN.match(series_id):
            logger.warning(
                "correlation_tracker_invalid_series_id",
                extra={
                    "event": "WARNING",
                    "reason": "series_id_validation_failed",
                    "series_id": series_id[:50],  # Truncate for safety
                    "action_taken": "reject_series_id",
                },
            )
            return False
        return True
    
    def update(
        self,
        series_id: str,
        value: float,
    ) -> None:
        """Update sliding window for a series.
        
        Args:
            series_id: Series identifier.
            value: New observation value.
        """
        # IEC 62443: Validate series_id
        if not self._validate_series_id(series_id):
            return
        
        if series_id not in self._data:
            self._data[series_id] = deque(maxlen=self._window_size)
        
        self._data[series_id].append(value)
    
    def get_correlated(
        self,
        series_id: str,
        threshold: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """Get series correlated with target series.
        
        Args:
            series_id: Target series identifier.
            threshold: Minimum absolute correlation.
        
        Returns:
            List of (series_id, correlation) tuples.
        """
        # IEC 62443: Validate series_id
        if not self._validate_series_id(series_id):
            return []
        
        if series_id not in self._data:
            return []
        
        target_data = list(self._data[series_id])
        
        if len(target_data) < self._min_samples:
            return []
        
        correlated = []
        
        for other_id, other_data in self._data.items():
            if other_id == series_id:
                continue
            
            if len(other_data) < self._min_samples:
                continue
            
            # Compute correlation
            correlation = self._compute_correlation(
                target_data,
                list(other_data),
            )
            
            if correlation is not None and abs(correlation) >= threshold:
                correlated.append((other_id, correlation))
        
        # Sort by absolute correlation descending
        correlated.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return correlated
    
    def _compute_correlation(
        self,
        x: List[float],
        y: List[float],
    ) -> Optional[float]:
        """Compute Pearson correlation between two series.
        
        Handles different lengths by using minimum common length.
        """
        try:
            # Use minimum common length
            n = min(len(x), len(y))
            if n < self._min_samples:
                return None
            
            x_arr = np.array(x[-n:])
            y_arr = np.array(y[-n:])
            
            # Compute Pearson correlation
            correlation = np.corrcoef(x_arr, y_arr)[0, 1]
            
            # Handle NaN (constant series)
            if np.isnan(correlation):
                return 0.0
            
            return float(correlation)
        
        except Exception as e:
            logger.debug(f"correlation_computation_failed: {e}")
            return None
    
    def get_window_data(self, series_id: str) -> Optional[List[float]]:
        """Get current window data for a series.
        
        Args:
            series_id: Series identifier.
        
        Returns:
            List of values or None if series not found.
        """
        # IEC 62443: Validate series_id
        if not self._validate_series_id(series_id):
            return None
        
        if series_id not in self._data:
            return None
        
        return list(self._data[series_id])
