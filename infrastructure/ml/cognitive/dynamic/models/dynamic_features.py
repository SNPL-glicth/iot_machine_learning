"""
DynamicFeatures dataclass for encapsulating dynamic feature computation results.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class DynamicFeatures:
    """Encapsulates dynamic operational features for a sensor reading."""
    
    sensor_id: int
    timestamp: float
    model_version: str = "2.0.0"
    
    # Derivatives
    derivative: Optional[float] = None
    second_derivative: Optional[float] = None
    
    # Rolling statistics (1h, 6h, 24h windows)
    rolling_mean_1h: Optional[float] = None
    rolling_std_1h: Optional[float] = None
    rolling_min_1h: Optional[float] = None
    rolling_max_1h: Optional[float] = None
    
    rolling_mean_6h: Optional[float] = None
    rolling_std_6h: Optional[float] = None
    rolling_min_6h: Optional[float] = None
    rolling_max_6h: Optional[float] = None
    
    rolling_mean_24h: Optional[float] = None
    rolling_std_24h: Optional[float] = None
    rolling_min_24h: Optional[float] = None
    rolling_max_24h: Optional[float] = None
    
    # Lag features
    lag_1: Optional[float] = None
    lag_6: Optional[float] = None
    lag_24: Optional[float] = None
    
    # Cross-features
    cross_features: Dict[str, float] = field(default_factory=dict)
    
    # Momentum (thermal, pressure gradient, etc.)
    thermal_momentum: Optional[float] = None
    pressure_gradient: Optional[float] = None
    
    # Volatility
    volatility_1h: Optional[float] = None
    volatility_6h: Optional[float] = None
    
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
            "rolling_min_1h": self.rolling_min_1h,
            "rolling_max_1h": self.rolling_max_1h,
            "rolling_mean_6h": self.rolling_mean_6h,
            "rolling_std_6h": self.rolling_std_6h,
            "rolling_min_6h": self.rolling_min_6h,
            "rolling_max_6h": self.rolling_max_6h,
            "rolling_mean_24h": self.rolling_mean_24h,
            "rolling_std_24h": self.rolling_std_24h,
            "rolling_min_24h": self.rolling_min_24h,
            "rolling_max_24h": self.rolling_max_24h,
            "lag_1": self.lag_1,
            "lag_6": self.lag_6,
            "lag_24": self.lag_24,
            "cross_features": self.cross_features,
            "thermal_momentum": self.thermal_momentum,
            "pressure_gradient": self.pressure_gradient,
            "volatility_1h": self.volatility_1h,
            "volatility_6h": self.volatility_6h,
        }
    
    def has_any_features(self) -> bool:
        """Check if any dynamic features were computed."""
        return (
            self.derivative is not None
            or self.second_derivative is not None
            or self.rolling_mean_1h is not None
            or self.lag_1 is not None
            or len(self.cross_features) > 0
        )
