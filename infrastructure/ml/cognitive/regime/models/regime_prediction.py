"""
RegimePrediction dataclass for encapsulating regime prediction results.
"""

from dataclasses import dataclass


@dataclass
class RegimePrediction:
    """Encapsulates regime prediction from classifier."""
    
    regime: str
    confidence: float
    cluster_idx: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "regime": self.regime,
            "confidence": self.confidence,
            "cluster_idx": self.cluster_idx,
        }
