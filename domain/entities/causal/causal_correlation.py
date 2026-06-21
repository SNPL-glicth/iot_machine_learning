"""
CausalCorrelation domain entity for operational causal relationships.

This is a domain entity for representing causal correlations between sensors.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class CausalCorrelation:
    """
    Domain entity for causal correlation (DDD value object).
    
    This represents the domain concept of causal correlation between sensors.
    """
    
    source_sensor_id: int
    target_sensor_id: int
    correlation_coefficient: float
    lag_seconds: float
    confidence: float
    propagation_likelihood: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "source_sensor_id": self.source_sensor_id,
            "target_sensor_id": self.target_sensor_id,
            "correlation_coefficient": self.correlation_coefficient,
            "lag_seconds": self.lag_seconds,
            "confidence": self.confidence,
            "propagation_likelihood": self.propagation_likelihood,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
