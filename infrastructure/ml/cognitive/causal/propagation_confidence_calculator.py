"""
PropagationConfidenceCalculator for calculating propagation confidence.

Implements confidence based on historical frequency, temporal consistency, contextual stability, and operational correlation.
"""

from typing import Dict, Any, List
import statistics


class PropagationConfidenceCalculator:
    """Calculator for propagation confidence."""
    
    def __init__(
        self,
        frequency_weight: float = 0.4,
        consistency_weight: float = 0.3,
        stability_weight: float = 0.2,
        correlation_weight: float = 0.1,
    ):
        """
        Initialize propagation confidence calculator.
        
        Args:
            frequency_weight: Weight for historical frequency
            consistency_weight: Weight for temporal consistency
            stability_weight: Weight for contextual stability
            correlation_weight: Weight for operational correlation
        """
        self._frequency_weight = frequency_weight
        self._consistency_weight = consistency_weight
        self._stability_weight = stability_weight
        self._correlation_weight = correlation_weight
    
    def calculate(
        self,
        historical_frequency: float,
        temporal_consistency: float,
        contextual_stability: float,
        operational_correlation: float,
    ) -> float:
        """
        Calculate propagation confidence.
        
        Args:
            historical_frequency: Historical frequency [0, 1]
            temporal_consistency: Temporal consistency [0, 1]
            contextual_stability: Contextual stability [0, 1]
            operational_correlation: Operational correlation [0, 1]
        
        Returns:
            Propagation confidence [0, 1]
        """
        confidence = (
            self._frequency_weight * historical_frequency +
            self._consistency_weight * temporal_consistency +
            self._stability_weight * contextual_stability +
            self._correlation_weight * operational_correlation
        )
        
        return min(1.0, max(0.0, confidence))
    
    def calculate_from_statistics(
        self,
        propagation_count: int,
        total_observations: int,
        duration_variance: float,
        context_match_rate: float,
        correlation_coefficient: float,
    ) -> float:
        """
        Calculate confidence from propagation statistics.
        
        Args:
            propagation_count: Number of propagations observed
            total_observations: Total number of observations
            duration_variance: Variance in propagation duration
            context_match_rate: Rate of context match [0, 1]
            correlation_coefficient: Correlation coefficient [-1, 1]
        
        Returns:
            Propagation confidence [0, 1]
        """
        # Calculate historical frequency
        historical_frequency = propagation_count / total_observations if total_observations > 0 else 0.0
        
        # Calculate temporal consistency (inverse of variance)
        temporal_consistency = max(0.0, 1.0 - min(1.0, duration_variance / 100.0))
        
        # Contextual stability
        contextual_stability = context_match_rate
        
        # Operational correlation (absolute value)
        operational_correlation = abs(correlation_coefficient)
        
        return self.calculate(
            historical_frequency,
            temporal_consistency,
            contextual_stability,
            operational_correlation,
        )
    
    def calculate_batch(
        self,
        statistics_list: List[Dict[str, Any]],
    ) -> List[float]:
        """
        Calculate confidence for multiple statistics.
        
        Args:
            statistics_list: List of statistics dictionaries
        
        Returns:
            List of confidence scores
        """
        confidences = []
        
        for stats in statistics_list:
            confidence = self.calculate_from_statistics(
                propagation_count=stats.get("propagation_count", 0),
                total_observations=stats.get("total_observations", 1),
                duration_variance=stats.get("duration_variance", 0.0),
                context_match_rate=stats.get("context_match_rate", 0.0),
                correlation_coefficient=stats.get("correlation_coefficient", 0.0),
            )
            confidences.append(confidence)
        
        return confidences
