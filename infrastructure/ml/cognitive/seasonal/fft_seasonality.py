"""FFT-based seasonality detector — lightweight cycle detection.

Reuses existing cycle_detector from engines/seasonal for consistency.
Provides decomposition interface compatible with STL.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class FFTSeasonalityDetector:
    """FFT-based seasonal component extraction.
    
    Detects dominant cycle using FFT and projects seasonal component.
    Lightweight alternative to STL when statsmodels unavailable.
    
    Attributes:
        _min_period: Minimum detectable cycle length.
        _max_period: Maximum detectable cycle length.
        _fft_threshold: Peak prominence threshold.
    """
    
    def __init__(
        self,
        min_period: int = 4,
        max_period: int = 100,
        fft_threshold: float = 0.15,
    ) -> None:
        """Initialize FFT seasonality detector.
        
        Args:
            min_period: Minimum detectable cycle length.
            max_period: Maximum detectable cycle length.
            fft_threshold: Peak prominence threshold.
        """
        self._min_period = min_period
        self._max_period = max_period
        self._fft_threshold = fft_threshold
    
    def decompose(
        self,
        values: list[float],
    ) -> Optional[Tuple[list[float], list[float], list[float]]]:
        """Decompose time series using FFT cycle detection.
        
        Args:
            values: Time series values.
        
        Returns:
            Tuple of (trend, seasonal, residual) or None if no cycle detected.
        """
        if len(values) < self._min_period * 2:
            return None
        
        try:
            # Import cycle detector from existing seasonal engine
            from iot_machine_learning.infrastructure.ml.engines.seasonal.cycle_detector import (
                detect_cycle,
            )
            from dataclasses import dataclass
            
            # Create config compatible with cycle_detector
            @dataclass
            class Config:
                min_period: int
                max_period: int
                fft_threshold: float
            
            config = Config(
                min_period=self._min_period,
                max_period=self._max_period,
                fft_threshold=self._fft_threshold,
            )
            
            # Detect cycle
            period, confidence = detect_cycle(values, config)
            
            if period is None or confidence < 0.3:
                return None
            
            # Project seasonal component using detected period
            seasonal = self._project_seasonal(values, period)
            
            # Compute trend as moving average
            trend = self._compute_trend(values, period)
            
            # Residual = values - trend - seasonal
            residual = [
                v - t - s
                for v, t, s in zip(values, trend, seasonal)
            ]
            
            return (trend, seasonal, residual)
        
        except Exception as e:
            logger.error(
                "fft_decomposition_failed",
                extra={
                    "event": "PHASE_ERROR",
                    "error": str(e),
                    "action_taken": "return_none",
                },
            )
            return None
    
    def _project_seasonal(
        self,
        values: list[float],
        period: int,
    ) -> list[float]:
        """Project seasonal component using detected period."""
        import numpy as np
        
        n = len(values)
        seasonal = []
        
        for i in range(n):
            # Position in cycle
            position = i % period
            
            # Collect values at same position from all cycles
            cycle_values = []
            for j in range(0, n, period):
                idx = j + position
                if idx < n:
                    cycle_values.append(values[idx])
            
            # Seasonal component = mean of cycle values
            if cycle_values:
                seasonal.append(float(np.mean(cycle_values)))
            else:
                seasonal.append(0.0)
        
        return seasonal
    
    def _compute_trend(
        self,
        values: list[float],
        period: int,
    ) -> list[float]:
        """Compute trend as moving average."""
        import numpy as np
        
        n = len(values)
        window = min(period, n)
        trend = []
        
        for i in range(n):
            start = max(0, i - window // 2)
            end = min(n, i + window // 2 + 1)
            window_values = values[start:end]
            trend.append(float(np.mean(window_values)))
        
        return trend
