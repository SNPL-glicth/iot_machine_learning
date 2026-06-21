"""
Correlation calculator utilities for causal correlation engine.

Provides Pearson correlation calculation and lagged correlation computation.
"""

from typing import List, Tuple
import statistics


class CorrelationCalculator:
    """Calculator for correlation metrics."""
    
    @staticmethod
    def pearson_correlation(x: List[float], y: List[float]) -> float:
        """Compute Pearson correlation coefficient."""
        n = len(x)
        if n < 2:
            return 0.0
        
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)
        
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denominator_x = sum((xi - mean_x) ** 2 for xi in x)
        denominator_y = sum((yi - mean_y) ** 2 for yi in y)
        
        if denominator_x == 0 or denominator_y == 0:
            return 0.0
        
        return numerator / (denominator_x * denominator_y) ** 0.5
    
    @staticmethod
    def compute_correlation_at_lag(
        source_data: List[Tuple[float, float]],
        target_data: List[Tuple[float, float]],
        lag_seconds: float,
    ) -> float:
        """Compute correlation at specific lag."""
        source_values = []
        target_values = []
        
        for source_ts, source_val in source_data:
            target_ts = source_ts + lag_seconds
            
            closest_target = min(
                target_data,
                key=lambda x: abs(x[0] - target_ts),
            )
            
            if abs(closest_target[0] - target_ts) < 5.0:
                source_values.append(source_val)
                target_values.append(closest_target[1])
        
        if len(source_values) < 5:
            return 0.0
        
        return CorrelationCalculator.pearson_correlation(source_values, target_values)
