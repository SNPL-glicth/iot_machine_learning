"""
RegimeState dataclass for tracking regime history.
"""

from dataclasses import dataclass


@dataclass
class RegimeState:
    """Value object for regime state at a point in time."""
    
    regime: str
    timestamp: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "regime": self.regime,
            "timestamp": self.timestamp,
        }
