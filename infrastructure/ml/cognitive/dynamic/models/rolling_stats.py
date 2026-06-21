"""
RollingStats value object for encapsulating rolling window statistics.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RollingStats:
    """Value object for rolling window statistics."""
    
    mean: float
    std: float
    min: float
    max: float
    count: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "count": self.count,
        }
