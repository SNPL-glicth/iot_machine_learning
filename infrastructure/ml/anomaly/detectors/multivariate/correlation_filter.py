"""Correlation Filter — dynamic series correlation tracking and filtering.

Extracted from MultivariateDetector for single responsibility.
Uses DynamicCorrelationTracker to select highly correlated series.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from iot_machine_learning.infrastructure.ml.engines.multivariate import (
    DynamicCorrelationTracker,
)

logger = logging.getLogger(__name__)


class CorrelationFilter:
    """Filters series by correlation strength.
    
    Attributes:
        tracker: Dynamic correlation tracker
        threshold: Minimum correlation coefficient
    """
    
    def __init__(
        self,
        window_size: int = 100,
        min_samples: int = 30,
        threshold: float = 0.5,
    ) -> None:
        """Initialize correlation filter.
        
        Args:
            window_size: Correlation window size
            min_samples: Minimum samples for correlation
            threshold: Minimum correlation coefficient
        """
        self._tracker = DynamicCorrelationTracker(
            window_size=window_size,
            min_samples=min_samples,
        )
        self._threshold = threshold
    
    def filter_by_correlation(
        self,
        series_id: str,
        target_values: List[float],
        candidate_series: Dict[str, List[float]],
    ) -> Dict[str, List[float]]:
        """Filter candidate series by correlation with target.
        
        Args:
            series_id: Target series ID
            target_values: Target series values
            candidate_series: Dict of {series_id: values}
        
        Returns:
            Filtered dict with only correlated series
        """
        if not candidate_series:
            return {}
        
        try:
            # Determine window size (minimum across all series)
            window_size = len(target_values)
            for series_values in candidate_series.values():
                window_size = min(window_size, len(series_values))
            
            if window_size == 0:
                logger.warning(
                    "correlation_filter_empty_window",
                    extra={
                        "series_id": series_id,
                        "action": "return_all_candidates",
                    },
                )
                return candidate_series
            
            # Update tracker with target series
            for val in target_values[-window_size:]:
                self._tracker.update(series_id, val)
            
            # Update tracker with candidate series
            for sid, svalues in candidate_series.items():
                for val in svalues[-window_size:]:
                    self._tracker.update(sid, val)
            
            # Get correlated series
            correlated = self._tracker.get_correlated(
                series_id,
                threshold=self._threshold,
            )
            
            if not correlated:
                logger.debug(
                    "correlation_filter_no_correlated",
                    extra={
                        "series_id": series_id,
                        "n_candidates": len(candidate_series),
                        "threshold": self._threshold,
                        "action": "return_all_candidates",
                    },
                )
                return candidate_series
            
            # Filter candidates
            correlated_ids = {sid for sid, _ in correlated}
            filtered = {
                sid: vals
                for sid, vals in candidate_series.items()
                if sid in correlated_ids
            }
            
            logger.debug(
                "correlation_filter_applied",
                extra={
                    "series_id": series_id,
                    "n_candidates": len(candidate_series),
                    "n_selected": len(filtered),
                    "top_correlations": [
                        {"series": sid, "corr": round(corr, 3)}
                        for sid, corr in correlated[:3]
                    ],
                },
            )
            
            return filtered
        
        except Exception as e:
            logger.warning(
                "correlation_filter_failed",
                extra={
                    "event": "FILTER_ERROR",
                    "error": str(e),
                    "action_taken": "use_all_series",
                },
            )
            return candidate_series
    
    def get_correlation_matrix(
        self,
        series_ids: List[str],
    ) -> Dict[Tuple[str, str], float]:
        """Get correlation matrix for series.
        
        Args:
            series_ids: List of series IDs
        
        Returns:
            Dict of {(series_a, series_b): correlation}
        """
        matrix = {}
        for i, sid_a in enumerate(series_ids):
            for sid_b in series_ids[i+1:]:
                correlated = self._tracker.get_correlated(sid_a, threshold=0.0)
                for sid, corr in correlated:
                    if sid == sid_b:
                        matrix[(sid_a, sid_b)] = corr
                        matrix[(sid_b, sid_a)] = corr
                        break
        return matrix
