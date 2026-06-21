"""
PropagationEvent domain entity for event propagation tracking.

This is a domain entity for representing operational propagation events.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass(frozen=True)
class PropagationEvent:
    """
    Domain entity for propagation event (DDD value object).
    
    This represents the domain concept of operational propagation event.
    """
    
    propagation_id: str
    source_sensor_id: int
    target_sensors: List[int]
    start_timestamp: float
    end_timestamp: float
    propagation_duration_seconds: float
    propagation_path: List[int]
    confidence: float
    is_cascade: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "propagation_id": self.propagation_id,
            "source_sensor_id": self.source_sensor_id,
            "target_sensors": self.target_sensors,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "propagation_duration_seconds": self.propagation_duration_seconds,
            "propagation_path": self.propagation_path,
            "confidence": self.confidence,
            "is_cascade": self.is_cascade,
            "metadata": self.metadata,
        }
