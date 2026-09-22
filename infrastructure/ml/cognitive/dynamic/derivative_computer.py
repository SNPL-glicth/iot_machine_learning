"""
Derivative calculator for dynamic feature computation.

Computes first and second order derivatives with smoothing for time-series data.
"""

from typing import Optional, List
import numpy as np


class DerivativeCalculator:
    """Calculates first and second order derivatives with optional smoothing."""
    
    def __init__(
        self,
        smoothing_window: int = 5,
        max_gap_seconds: float = 300.0,
    ):
        """
        Initialize derivative calculator.
        
        Args:
            smoothing_window: Number of recent points to average for smoothing (0 = no smoothing)
            max_gap_seconds: Maximum allowed gap between timestamps (seconds). 
                            Derivatives are None if gap exceeds this.
        """
        self._smoothing_window = smoothing_window
        self._max_gap_seconds = max_gap_seconds
    
    def compute_first_derivative(
        self,
        values: List[float],
        timestamps: List[float],
    ) -> Optional[float]:
        """
        Compute first derivative (Δ/Δt).
        
        Args:
            values: List of recent values
            timestamps: List of corresponding timestamps (Unix epoch)
        
        Returns:
            First derivative or None if insufficient data or gap too large
        """
        if len(values) < 2 or len(timestamps) < 2:
            return None
        
        n = len(values)
        dt = timestamps[-1] - timestamps[-2]
        
        # Validate gap
        if dt == 0 or dt > self._max_gap_seconds:
            return None
        
        dy = values[-1] - values[-2]
        derivative = dy / dt
        
        # Apply smoothing if enabled
        if self._smoothing_window > 1 and n >= self._smoothing_window:
            recent_derivatives = []
            for i in range(max(0, n - self._smoothing_window), n - 1):
                dt_i = timestamps[i + 1] - timestamps[i]
                if dt_i > 0 and dt_i < self._max_gap_seconds:
                    recent_derivatives.append((values[i + 1] - values[i]) / dt_i)
            
            if recent_derivatives:
                derivative = sum(recent_derivatives) / len(recent_derivatives)
        
        return derivative
    
    def compute_second_derivative(
        self,
        values: List[float],
        timestamps: List[float],
    ) -> Optional[float]:
        """
        Compute second derivative (d²/dt²).
        
        Args:
            values: List of recent values
            timestamps: List of corresponding timestamps (Unix epoch)
        
        Returns:
            Second derivative or None if insufficient data or gap too large
        """
        if len(values) < 3 or len(timestamps) < 3:
            return None
        
        # Compute first derivatives
        first_derivatives = []
        for i in range(len(values) - 1):
            dt = timestamps[i + 1] - timestamps[i]
            if dt > 0 and dt < self._max_gap_seconds:
                first_derivatives.append((values[i + 1] - values[i]) / dt)
        
        if len(first_derivatives) < 2:
            return None
        
        # Compute derivative of first derivatives
        dt = timestamps[-1] - timestamps[-3]  # Approximation
        if dt == 0:
            return None
        
        d2y = first_derivatives[-1] - first_derivatives[-2]
        second_derivative = d2y / dt
        
        return second_derivative
