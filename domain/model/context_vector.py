"""ContextVector — immutable context representation for MoE routing.

Pure domain object with no external dependencies.
Used by gating networks to make routing decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass(frozen=True)
class ContextVector:
    """Immutable vector of context features for MoE routing decisions.
    
    Contains all information necessary for a gating network to decide
    which experts to activate for a given prediction context.
    
    Attributes:
        regime: Detected signal regime (stable, trending, volatile, noisy).
        domain: Application domain (iot, finance, healthcare, etc.).
        n_points: Number of data points in the current window.
        signal_features: Extracted signal features (mean, std, slope, etc.).
        temporal_features: Optional temporal context (hour, day_of_week).
        metadata: Optional additional context metadata.
    
    Example:
        >>> ctx = ContextVector(
        ...     regime="volatile",
        ...     domain="iot",
        ...     n_points=10,
        ...     signal_features={"mean": 20.0, "std": 1.5, "slope": 0.5}
        ... )
        >>> arr = ctx.to_array()
        >>> print(len(arr))  # Deterministic length
    """
    
    regime: str
    domain: str
    n_points: int
    signal_features: Dict[str, float]
    temporal_features: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate context vector invariants."""
        # Frozen dataclass requires object.__setattr__ for validation
        if self.n_points < 0:
            raise ValueError(f"n_points must be non-negative, got {self.n_points}")
        if not self.regime:
            raise ValueError("regime cannot be empty")
        if not self.domain:
            raise ValueError("domain cannot be empty")
    
    def to_array(self) -> List[float]:
        """Convert context to numeric array for ML models.
        
        Returns a deterministic list of floats in consistent order.
        Suitable for neural network or tree-based gating inputs.
        
        Order:
        1. n_points (float)
        2. Core signal features: mean, std, slope, curvature, noise_ratio, stability
        3. Regime one-hot encoding (stable, trending, volatile, noisy)
        
        Returns:
            List of float values.
        """
        # Core numeric features
        base_features = [
            float(self.n_points),
            self.signal_features.get("mean", 0.0),
            self.signal_features.get("std", 0.0),
            self.signal_features.get("slope", 0.0),
            self.signal_features.get("curvature", 0.0),
            self.signal_features.get("noise_ratio", 0.0),
            self.signal_features.get("stability", 0.0),
        ]
        
        # Regime one-hot encoding (deterministic order)
        regimes = ["stable", "trending", "volatile", "noisy"]
        regime_encoded = [1.0 if r == self.regime else 0.0 for r in regimes]
        
        return base_features + regime_encoded
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.
        
        Returns:
            Dictionary representation of the context.
        """
        return {
            "regime": self.regime,
            "domain": self.domain,
            "n_points": self.n_points,
            "signal_features": dict(self.signal_features),
            "temporal_features": self.temporal_features,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextVector":
        """Create ContextVector from dictionary.
        
        Args:
            data: Dictionary with context data.
            
        Returns:
            New ContextVector instance.
        """
        return cls(
            regime=data["regime"],
            domain=data["domain"],
            n_points=data["n_points"],
            signal_features=data.get("signal_features", {}),
            temporal_features=data.get("temporal_features"),
            metadata=data.get("metadata"),
        )
    
    def with_regime(self, regime: str) -> "ContextVector":
        """Create new ContextVector with updated regime.
        
        Args:
            regime: New regime value.
            
        Returns:
            New immutable ContextVector.
        """
        return ContextVector(
            regime=regime,
            domain=self.domain,
            n_points=self.n_points,
            signal_features=dict(self.signal_features),
            temporal_features=self.temporal_features,
            metadata=self.metadata,
        )
    
    def with_signal_features(self, **features) -> "ContextVector":
        """Create new ContextVector with merged signal features.
        
        Args:
            **features: Additional or updated signal features.
            
        Returns:
            New immutable ContextVector.
        """
        merged = {**self.signal_features, **features}
        return ContextVector(
            regime=self.regime,
            domain=self.domain,
            n_points=self.n_points,
            signal_features=merged,
            temporal_features=self.temporal_features,
            metadata=self.metadata,
        )
    
    @property
    def feature_count(self) -> int:
        """Return the number of features in to_array()."""
        return 7 + 4  # base + regime one-hot
    
    @property
    def is_stable(self) -> bool:
        """True if regime is stable."""
        return self.regime == "stable"
    
    def has_minimum_points(self, min_points: int = 3) -> bool:
        """Check if context has minimum required points.
        
        Args:
            min_points: Minimum required data points.
            
        Returns:
            True if n_points >= min_points.
        """
        return self.n_points >= min_points
