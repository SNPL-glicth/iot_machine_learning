"""
FeatureConfig dataclass for configuring dynamic features per sensor type.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for dynamic features per sensor type."""
    
    # Enable/disable specific features
    enable_derivatives: bool = True
    enable_rolling_stats: bool = True
    enable_lag_features: bool = True
    enable_cross_features: bool = False
    enable_momentum: bool = False
    enable_gradient: bool = False
    enable_volatility: bool = False
    
    # Rolling window sizes (in minutes)
    rolling_windows: List[int] = field(default_factory=lambda: [60, 360, 1440])  # 1h, 6h, 24h
    
    # Lag periods
    lag_periods: List[int] = field(default_factory=lambda: [1, 6, 24])
    
    # Derivative settings
    derivative_smoothing_window: int = 5
    max_gap_seconds: float = 300.0
    
    # Cross-feature settings
    correlation_threshold: float = 0.7
    max_cross_features: int = 5
    
    # Limits to prevent dimensionality explosion
    max_total_features: int = 50
    max_rolling_windows: int = 3
    max_lag_periods: int = 3
    max_cross_features_per_sensor: int = 5
    
    @classmethod
    def default(cls) -> 'FeatureConfig':
        """Create default configuration."""
        return cls()
    
    @classmethod
    def for_temperature(cls) -> 'FeatureConfig':
        """Configuration optimized for temperature sensors."""
        return cls(
            enable_derivatives=True,
            enable_rolling_stats=True,
            enable_lag_features=True,
            enable_momentum=True,  # Thermal momentum
            enable_gradient=False,
            enable_volatility=True,
            rolling_windows=[60, 360, 1440],
            lag_periods=[1, 6, 24],
        )
    
    @classmethod
    def for_pressure(cls) -> 'FeatureConfig':
        """Configuration optimized for pressure sensors."""
        return cls(
            enable_derivatives=True,
            enable_rolling_stats=True,
            enable_lag_features=True,
            enable_momentum=False,
            enable_gradient=True,  # Pressure gradient
            enable_volatility=True,
            rolling_windows=[60, 360],
            lag_periods=[1, 6],
        )
    
    @classmethod
    def for_vibration(cls) -> 'FeatureConfig':
        """Configuration optimized for vibration sensors."""
        return cls(
            enable_derivatives=True,
            enable_rolling_stats=True,
            enable_lag_features=True,
            enable_momentum=False,
            enable_gradient=False,
            enable_volatility=True,
            rolling_windows=[30, 60],  # Shorter windows
            lag_periods=[1, 6],
        )
    
    @classmethod
    def minimal(cls) -> 'FeatureConfig':
        """Minimal configuration for quick wins."""
        return cls(
            enable_derivatives=True,
            enable_rolling_stats=True,
            enable_lag_features=False,
            enable_cross_features=False,
            enable_momentum=False,
            enable_gradient=False,
            enable_volatility=False,
            rolling_windows=[60],  # Only 1h
            lag_periods=[],
        )
