"""STL decomposer — Seasonal-Trend decomposition using LOESS.

Wrapper around statsmodels STL for seasonal component extraction.
Gracefully falls back to None if statsmodels not available.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class STLDecomposer:
    """STL-based seasonal decomposition.
    
    Extracts seasonal component using LOESS smoothing.
    Requires statsmodels package.
    
    Attributes:
        _period: Seasonal period length.
        _seasonal: Seasonal smoother parameter (odd integer).
    """
    
    def __init__(
        self,
        period: int,
        seasonal: int = 7,
    ) -> None:
        """Initialize STL decomposer.
        
        Args:
            period: Length of seasonal cycle.
            seasonal: Seasonal smoother parameter (must be odd).
        
        Raises:
            ValueError: If period < 2 or seasonal is even.
        """
        if period < 2:
            raise ValueError(f"period must be >= 2, got {period}")
        if seasonal % 2 == 0:
            raise ValueError(f"seasonal must be odd, got {seasonal}")
        
        self._period = period
        self._seasonal = seasonal
        self._available = self._check_statsmodels()
    
    def _check_statsmodels(self) -> bool:
        """Check if statsmodels is available."""
        try:
            import statsmodels.api
            return True
        except ImportError:
            logger.warning(
                "stl_decomposer_statsmodels_unavailable",
                extra={
                    "event": "WARNING",
                    "reason": "statsmodels_not_installed",
                    "action_taken": "stl_decomposer_disabled",
                },
            )
            return False
    
    @property
    def available(self) -> bool:
        """True if statsmodels is available."""
        return self._available
    
    def decompose(
        self,
        values: list[float],
    ) -> Optional[Tuple[list[float], list[float], list[float]]]:
        """Decompose time series into trend, seasonal, and residual.
        
        Args:
            values: Time series values.
        
        Returns:
            Tuple of (trend, seasonal, residual) or None if unavailable.
        """
        if not self._available:
            return None
        
        if len(values) < self._period * 2:
            logger.debug(
                "stl_insufficient_data",
                extra={
                    "required": self._period * 2,
                    "received": len(values),
                },
            )
            return None
        
        try:
            from statsmodels.tsa.seasonal import STL
            import numpy as np
            
            # Run STL decomposition
            stl = STL(
                np.array(values),
                period=self._period,
                seasonal=self._seasonal,
            )
            result = stl.fit()
            
            return (
                result.trend.tolist(),
                result.seasonal.tolist(),
                result.resid.tolist(),
            )
        
        except Exception as e:
            logger.error(
                "stl_decomposition_failed",
                extra={
                    "event": "PHASE_ERROR",
                    "error": str(e),
                    "action_taken": "return_none",
                },
            )
            return None
