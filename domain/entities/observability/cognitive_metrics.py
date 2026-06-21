"""
CognitiveMetrics domain entity for cognitive observability.

This is a domain entity for collecting and storing cognitive metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime


@dataclass(frozen=True)
class CognitiveMetrics:
    """
    Domain entity for cognitive metrics (DDD value object).
    
    This represents the domain concept of cognitive metrics for observability.
    """
    
    timestamp: float
    regime_distribution: Dict[str, int]
    anomaly_distribution: Dict[str, int]
    retrieval_hit_rate: float
    retrieval_similarity_mean: float
    explainability_consistency: float
    confidence_distribution: Dict[str, float]
    memory_growth_rate: float
    ttl_cleanup_rate: float
    contextual_confidence_calibration: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "regime_distribution": self.regime_distribution,
            "anomaly_distribution": self.anomaly_distribution,
            "retrieval_hit_rate": self.retrieval_hit_rate,
            "retrieval_similarity_mean": self.retrieval_similarity_mean,
            "explainability_consistency": self.explainability_consistency,
            "confidence_distribution": self.confidence_distribution,
            "memory_growth_rate": self.memory_growth_rate,
            "ttl_cleanup_rate": self.ttl_cleanup_rate,
            "contextual_confidence_calibration": self.contextual_confidence_calibration,
        }
