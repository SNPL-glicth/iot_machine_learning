"""
Domain entity for dynamic features following DDD principles.

This is a value object representing dynamic operational features in the domain layer.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class DomainDynamicFeatures:
    """
    Domain entity for dynamic features (DDD value object).
    
    This represents the domain concept of dynamic operational features,
    separate from infrastructure concerns.
    """
    
    sensor_id: int
    timestamp: float
    model_version: str = "2.0.0"
    
    # Derivatives
    derivative: Optional[float] = None
    second_derivative: Optional[float] = None
    
    # Rolling statistics (simplified for domain)
    rolling_mean_1h: Optional[float] = None
    rolling_std_1h: Optional[float] = None
    
    # Lag features
    lag_1: Optional[float] = None
    lag_6: Optional[float] = None
    
    # Cross-features
    cross_features: Dict[str, float] = None
    
    def __post_init__(self):
        if self.cross_features is None:
            object.__setattr__(self, 'cross_features', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sensor_id": self.sensor_id,
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "derivative": self.derivative,
            "second_derivative": self.second_derivative,
            "rolling_mean_1h": self.rolling_mean_1h,
            "rolling_std_1h": self.rolling_std_1h,
            "lag_1": self.lag_1,
            "lag_6": self.lag_6,
            "cross_features": self.cross_features,
        }
