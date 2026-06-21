"""
Granger causality utilities for causal correlation engine.

Provides simplified Granger causality detection.
"""

from typing import List, Tuple
import statistics


class GrangerCausalityDetector:
    """Detector for Granger causality."""
    
    @staticmethod
    def calculate_prediction_error(data: List[Tuple[float, float]]) -> float:
        """Calculate prediction error (mean squared error)."""
        if len(data) < 2:
            return 0.0
        
        values = [val for _, val in data]
        mean_val = statistics.mean(values)
        
        error = sum((val - mean_val) ** 2 for val in values) / len(values)
        return error
    
    @staticmethod
    def calculate_error_with_predictor(
        target_data: List[Tuple[float, float]],
        source_data: List[Tuple[float, float]],
        lag: int,
    ) -> float:
        """Calculate prediction error using source as predictor."""
        if len(target_data) < lag + 1:
            return float('inf')
        
        errors = []
        for i in range(lag, len(target_data)):
            target_val = target_data[i][1]
            source_val = source_data[i - lag][1]
            predicted_val = source_val
            error = (target_val - predicted_val) ** 2
            errors.append(error)
        
        return statistics.mean(errors) if errors else float('inf')
    
    @staticmethod
    def detect_granger_causality(
        source_data: List[Tuple[float, float]],
        target_data: List[Tuple[float, float]],
        max_lag: int = 10,
    ) -> float:
        """Detect basic Granger causality (simplified)."""
        if len(source_data) < max_lag + 10 or len(target_data) < max_lag + 10:
            return 0.0
        
        base_error = GrangerCausalityDetector.calculate_prediction_error(target_data)
        error_with_source = GrangerCausalityDetector.calculate_error_with_predictor(
            target_data, source_data, max_lag
        )
        
        if base_error == 0:
            return 1.0
        
        causality_score = (base_error - error_with_source) / base_error
        return max(0.0, min(1.0, causality_score))
