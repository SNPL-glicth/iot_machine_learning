"""
MemoryEvent domain entity following DDD principles.

This is a value object representing operational memory events in the domain layer.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class MemoryEvent:
    """
    Domain entity for operational memory event (DDD value object).
    
    This represents the domain concept of an operational memory event,
    separate from infrastructure concerns.
    """
    
    series_id: int
    series_type: str
    timestamp: float
    event_type: str
    semantic_text: str
    regime: str
    anomaly_score: float
    dynamic_features: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "series_id": self.series_id,
            "series_type": self.series_type,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "semantic_text": self.semantic_text,
            "regime": self.regime,
            "anomaly_score": self.anomaly_score,
            "dynamic_features": self.dynamic_features,
            "metadata": self.metadata,
        }