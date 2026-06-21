"""
ContextualConfidenceCalculator for calculating contextual confidence.

Implements hybrid confidence calculation with weighting:
- anomaly_score
- retrieval similarity
- regime confidence
- feature stability
"""

from typing import Dict, Any


class ContextualConfidenceCalculator:
    """Calculator for contextual confidence."""
    
    def __init__(
        self,
        anomaly_weight: float = 0.5,
        retrieval_weight: float = 0.3,
        regime_weight: float = 0.15,
        stability_weight: float = 0.05,
    ):
        """
        Initialize confidence calculator.
        
        Args:
            anomaly_weight: Weight for anomaly score
            retrieval_weight: Weight for retrieval similarity
            regime_weight: Weight for regime confidence
            stability_weight: Weight for feature stability
        """
        self._anomaly_weight = anomaly_weight
        self._retrieval_weight = retrieval_weight
        self._regime_weight = regime_weight
        self._stability_weight = stability_weight
    
    def calculate(
        self,
        anomaly_score: float,
        retrieval_similarity: float,
        regime_confidence: float,
        feature_stability: float,
    ) -> float:
        """
        Calculate contextual confidence.
        
        Args:
            anomaly_score: Anomaly score [0, 1]
            retrieval_similarity: Retrieval similarity [0, 1]
            regime_confidence: Regime classification confidence [0, 1]
            feature_stability: Feature stability [0, 1]
        
        Returns:
            Contextual confidence [0, 1]
        """
        confidence = (
            self._anomaly_weight * anomaly_score +
            self._retrieval_weight * retrieval_similarity +
            self._regime_weight * regime_confidence +
            self._stability_weight * feature_stability
        )
        
        return min(1.0, max(0.0, confidence))
