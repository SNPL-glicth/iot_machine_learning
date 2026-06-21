"""
OperationalRegime domain entity following DDD principles.

This is a value object representing operational regimes in the domain layer.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OperationalRegime:
    """
    Domain entity for operational regime (DDD value object).
    
    This represents the domain concept of operational regime,
    separate from infrastructure concerns.
    """
    
    name: str
    confidence: float
    timestamp: float
    sensor_id: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "sensor_id": self.sensor_id,
        }
