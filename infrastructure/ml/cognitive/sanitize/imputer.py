"""Value imputation strategies for SanitizePhase (COG-CRIT-1).

Imputation replaces invalid values (NaN, Inf) with estimated values from history
instead of rejecting the entire window.

Applies SRP: Imputer only imputes, no other concerns.
Applies OCP: Abstract Imputer allows interchangeable strategies (median, mean, forward fill).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Optional


class Imputer(ABC):
    """Abstract imputation strategy for invalid values (COG-CRIT-1).
    
    Applies OCP: Different strategies (median, mean, forward fill) can be
    implemented without modifying SanitizePhase.
    """
    
    @abstractmethod
    def impute(self, value: float, history: List[float]) -> float:
        """Impute invalid value using historical data.
        
        Args:
            value: Invalid value to impute (NaN or Inf).
            history: Historical values for estimation.
        
        Returns:
            Imputed value.
        
        Raises:
            ValueError: If history is insufficient for imputation.
        
        Applies SRP: Only imputes, no validation or logging.
        """
        pass


class MedianImputer(Imputer):
    """Median-based imputation strategy.
    
    Replaces invalid values with median of historical data.
    Robust to outliers and works well for skewed distributions.
    
    Attributes:
        min_history: Minimum historical values required for imputation.
        fallback_value: Value to use if history is insufficient.
    """
    
    def __init__(
        self,
        min_history: int = 3,
        fallback_value: Optional[float] = None,
    ) -> None:
        """Initialize median imputer.
        
        Args:
            min_history: Minimum values required for imputation.
            fallback_value: Fallback if history insufficient (None = raise error).
        """
        self._min_history = min_history
        self._fallback_value = fallback_value
    
    def impute(self, value: float, history: List[float]) -> float:
        """Impute using median of historical data.
        
        Args:
            value: Invalid value (may be NaN or Inf).
            history: Historical values for estimation.
        
        Returns:
            Median of history if sufficient, otherwise fallback.
        
        Raises:
            ValueError: If history insufficient and no fallback provided.
        """
        if not history or len(history) < self._min_history:
            if self._fallback_value is not None:
                return self._fallback_value
            raise ValueError(
                f"Insufficient history for median imputation: "
                f"need {self._min_history}, got {len(history)}"
            )
        
        # Filter out non-finite values from history
        valid_history = [h for h in history if math.isfinite(h)]
        if not valid_history:
            if self._fallback_value is not None:
                return self._fallback_value
            raise ValueError("No valid finite values in history for imputation")
        
        # Calculate median
        sorted_history = sorted(valid_history)
        n = len(sorted_history)
        if n % 2 == 0:
            median = (sorted_history[n // 2 - 1] + sorted_history[n // 2]) / 2.0
        else:
            median = sorted_history[n // 2]
        
        return median


class MeanImputer(Imputer):
    """Mean-based imputation strategy.
    
    Replaces invalid values with mean of historical data.
    Sensitive to outliers but computationally efficient.
    
    Attributes:
        min_history: Minimum historical values required for imputation.
        fallback_value: Value to use if history is insufficient.
    """
    
    def __init__(
        self,
        min_history: int = 3,
        fallback_value: Optional[float] = None,
    ) -> None:
        """Initialize mean imputer.
        
        Args:
            min_history: Minimum values required for imputation.
            fallback_value: Fallback if history insufficient (None = raise error).
        """
        self._min_history = min_history
        self._fallback_value = fallback_value
    
    def impute(self, value: float, history: List[float]) -> float:
        """Impute using mean of historical data."""
        if not history or len(history) < self._min_history:
            if self._fallback_value is not None:
                return self._fallback_value
            raise ValueError(
                f"Insufficient history for mean imputation: "
                f"need {self._min_history}, got {len(history)}"
            )
        
        valid_history = [h for h in history if math.isfinite(h)]
        if not valid_history:
            if self._fallback_value is not None:
                return self._fallback_value
            raise ValueError("No valid finite values in history for imputation")
        
        return sum(valid_history) / len(valid_history)
