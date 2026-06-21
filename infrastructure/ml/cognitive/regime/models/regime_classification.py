"""
RegimeClassification dataclass for encapsulating regime detection results.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RegimeClassification:
    """Encapsulates regime classification result with context."""
    
    sensor_id: int
    timestamp: float
    regime: str
    confidence: float
    previous_regime: Optional[str] = None
    transition_duration: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "sensor_id": self.sensor_id,
            "timestamp": self.timestamp,
            "regime": self.regime,
            "confidence": self.confidence,
            "previous_regime": self.previous_regime,
            "transition_duration": self.transition_duration,
        }
