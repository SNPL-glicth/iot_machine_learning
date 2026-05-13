"""Adaptive Inhibition Configuration (MATH-SEV-4 / SEVERO-1).

Calculates thresholds dynamically from historical error percentiles instead of hardcoded values.

Applies SRP: Threshold calculation is separate from inhibition gate logic.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .gate import InhibitionConfig

logger = logging.getLogger(__name__)


class AdaptiveInhibitionConfig(InhibitionConfig):
    """Adaptive inhibition config that calculates thresholds from historical errors.
    
    Extends InhibitionConfig to compute thresholds from percentiles of observed errors
    instead of using hardcoded values.
    
    Attributes:
        _error_history: Per-series error history for percentile calculation.
        _percentile: Percentile to use for threshold (default: 75th).
        _min_samples: Minimum samples required for adaptive calculation.
    
    Applies SRP: Only calculates thresholds, does not make inhibition decisions.
    Applies OCP: Extends base config without modifying it.
    """
    
    def __init__(
        self,
        percentile: float = 75.0,
        min_samples: int = 20,
        **kwargs,
    ) -> None:
        """Initialize adaptive config.
        
        Args:
            percentile: Percentile for threshold calculation (0-100).
            min_samples: Minimum samples required for adaptive thresholds.
            **kwargs: Base InhibitionConfig parameters.
        """
        super().__init__(**kwargs)
        
        if not (0 <= percentile <= 100):
            raise ValueError(f"percentile must be in [0, 100], got {percentile}")
        if min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {min_samples}")
        
        self._percentile = percentile
        self._min_samples = min_samples
        self._error_history: Dict[str, List[float]] = {}
    
    def update_error_history(
        self,
        series_id: str,
        errors: List[float],
    ) -> None:
        """Update error history for a series.
        
        Args:
            series_id: Series identifier.
            errors: Recent errors to add to history.
        
        Applies SRP: Only updates history, no threshold calculation.
        """
        if series_id not in self._error_history:
            self._error_history[series_id] = []
        
        self._error_history[series_id].extend(errors)
        
        # Keep only last 1000 errors to prevent unbounded growth
        if len(self._error_history[series_id]) > 1000:
            self._error_history[series_id] = self._error_history[series_id][-1000:]
    
    def get_adaptive_thresholds(
        self,
        series_id: str,
    ) -> Optional[tuple[float, float, float]]:
        """Calculate adaptive thresholds from historical errors.
        
        Args:
            series_id: Series identifier.
        
        Returns:
            Tuple of (fit_error_threshold, recent_error_threshold, stability_threshold)
            or None if insufficient data.
        
        Applies SRP: Only calculates thresholds from history.
        """
        errors = self._error_history.get(series_id, [])
        
        if len(errors) < self._min_samples:
            logger.debug(
                "adaptive_thresholds_insufficient_data",
                extra={
                    "series_id": series_id,
                    "n_errors": len(errors),
                    "min_samples": self._min_samples,
                },
            )
            return None
        
        # Calculate percentile
        sorted_errors = sorted(errors)
        percentile_idx = int(len(sorted_errors) * (self._percentile / 100.0))
        percentile_value = sorted_errors[percentile_idx]
        
        # Use percentile as threshold for fit_error and recent_error
        # Stability threshold remains fixed (it's not error-based)
        fit_error_threshold = max(1.0, percentile_value)  # Minimum 1.0
        recent_error_threshold = max(1.0, percentile_value)
        
        logger.info(
            "adaptive_thresholds_calculated",
            extra={
                "series_id": series_id,
                "percentile": self._percentile,
                "n_errors": len(errors),
                "fit_error_threshold": round(fit_error_threshold, 3),
                "recent_error_threshold": round(recent_error_threshold, 3),
            },
        )
        
        return (
            fit_error_threshold,
            recent_error_threshold,
            self.stability_threshold,  # Keep base stability threshold
        )
    
    def apply_adaptive_thresholds(
        self,
        series_id: str,
    ) -> None:
        """Apply adaptive thresholds to this config instance.
        
        Args:
            series_id: Series identifier.
        
        Mutates self.fit_error_threshold and self.recent_error_threshold if
        sufficient data is available.
        
        Applies SRP: Only updates thresholds, no inhibition logic.
        """
        thresholds = self.get_adaptive_thresholds(series_id)
        
        if thresholds is not None:
            fit_error, recent_error, stability = thresholds
            self.fit_error_threshold = fit_error
            self.recent_error_threshold = recent_error
            self.stability_threshold = stability
    
    def get_metrics(self, series_id: Optional[str] = None) -> Dict[str, any]:
        """Get adaptive config metrics.
        
        Args:
            series_id: Optional series to get specific metrics for.
        
        Returns:
            Dict with config state and metrics.
        """
        metrics = {
            "percentile": self._percentile,
            "min_samples": self._min_samples,
            "total_series_tracked": len(self._error_history),
        }
        
        if series_id and series_id in self._error_history:
            errors = self._error_history[series_id]
            metrics["series_error_count"] = len(errors)
            
            thresholds = self.get_adaptive_thresholds(series_id)
            if thresholds:
                metrics["adaptive_fit_error_threshold"] = thresholds[0]
                metrics["adaptive_recent_error_threshold"] = thresholds[1]
        
        return metrics
