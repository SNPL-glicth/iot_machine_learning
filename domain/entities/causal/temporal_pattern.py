"""
TemporalPattern domain entity for temporal pattern mining.

This is a domain entity for representing temporal operational patterns.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass(frozen=True)
class TemporalPattern:
    """
    Domain entity for temporal pattern (DDD value object).
    
    This represents the domain concept of temporal operational pattern.
    """
    
    pattern_id: str
    sequence: List[int]  # sensor IDs in sequence
    frequency: int
    avg_duration_seconds: float
    confidence: float
    is_pre_anomaly: bool
    timestamp: float
    pattern_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "pattern_id": self.pattern_id,
            "sequence": self.sequence,
            "frequency": self.frequency,
            "avg_duration_seconds": self.avg_duration_seconds,
            "confidence": self.confidence,
            "is_pre_anomaly": self.is_pre_anomaly,
            "timestamp": self.timestamp,
            "pattern_metadata": self.pattern_metadata,
        }
