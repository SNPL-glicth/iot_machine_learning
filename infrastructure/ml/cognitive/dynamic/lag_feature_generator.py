"""
Lag feature generator for dynamic feature computation.

Generates lag features (t-1, t-6, t-24) for time-series data.
"""

from typing import Dict, List, Optional


class LagFeatureGenerator:
    """Generates lag features for specified periods."""
    
    def __init__(self, default_lag_periods: Optional[List[int]] = None):
        """
        Initialize lag feature generator.
        
        Args:
            default_lag_periods: Default lag periods to compute (e.g., [1, 6, 24])
        """
        self._default_lag_periods = default_lag_periods or [1, 6, 24]
    
    def compute_lags(
        self,
        values: List[float],
        lag_periods: Optional[List[int]] = None,
    ) -> Dict[int, Optional[float]]:
        """
        Compute lag features for specified periods.
        
        Args:
            values: List of recent values (most recent last)
            lag_periods: Lag periods to compute (e.g., [1, 6, 24]).
                        If None, uses default_lag_periods.
        
        Returns:
            Dictionary mapping lag period to lagged value.
            Returns None for periods where insufficient data exists.
        """
        periods = lag_periods or self._default_lag_periods
        lags = {}
        
        for period in periods:
            if len(values) > period:
                lags[period] = values[-(period + 1)]
            else:
                lags[period] = None
        
        return lags
    
    def compute_single_lag(
        self,
        values: List[float],
        period: int,
    ) -> Optional[float]:
        """
        Compute a single lag feature.
        
        Args:
            values: List of recent values (most recent last)
            period: Lag period (e.g., 1 for t-1, 6 for t-6)
        
        Returns:
            Lagged value or None if insufficient data
        """
        if len(values) > period:
            return values[-(period + 1)]
        return None
