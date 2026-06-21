"""
RegimeConfig dataclass for configuring regime detection per sensor type.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RegimeConfig:
    """Configuration for regime detection per sensor type."""
    
    # Umbrales de derivada
    derivative_threshold_low: float = 1.0  # σ
    derivative_threshold_high: float = 3.0  # σ
    
    # Umbrales de volatilidad
    volatility_threshold_low: float = 0.5  # σ
    volatility_threshold_high: float = 2.0  # σ
    
    # Umbrales de valor
    low_threshold: float = -2.0  # σ desde baseline
    high_threshold: float = 2.0  # σ desde baseline
    
    # Duración mínima de régimen
    min_regime_duration: float = 300.0  # segundos
    
    # Features a usar
    use_derivative: bool = True
    use_rolling_std: bool = True
    use_second_derivative: bool = False
    use_lag_features: bool = False
    
    # Thresholds de velocity Z por régimen
    velocity_z_lower_startup: float = 3.0
    velocity_z_upper_startup: float = 5.0
    velocity_z_lower_shutdown: float = -5.0
    velocity_z_upper_shutdown: float = -3.0
    velocity_z_lower_stable: float = 2.0
    velocity_z_upper_stable: float = 3.0
    
    # Thresholds de acceleration Z por régimen
    acceleration_z_lower_startup: float = 3.0
    acceleration_z_upper_startup: float = 5.0
    acceleration_z_lower_shutdown: float = -5.0
    acceleration_z_upper_shutdown: float = -3.0
    acceleration_z_lower_stable: float = 2.0
    acceleration_z_upper_stable: float = 3.0
    
    # Multiplicadores de anomaly scoring por régimen
    peak_load_multiplier: float = 0.7
    transition_multiplier: float = 1.2
    startup_multiplier: float = 0.8
    normal_multiplier: float = 1.0
    
    @classmethod
    def default(cls) -> 'RegimeConfig':
        """Create default configuration."""
        return cls()
    
    @classmethod
    def for_temperature(cls) -> 'RegimeConfig':
        """Configuration optimized for temperature sensors."""
        return cls(
            derivative_threshold_low=1.0,
            derivative_threshold_high=3.0,
            volatility_threshold_low=0.5,
            volatility_threshold_high=2.0,
            min_regime_duration=300.0,
            use_derivative=True,
            use_rolling_std=True,
            use_second_derivative=False,
            use_lag_features=False,
        )
    
    @classmethod
    def for_pressure(cls) -> 'RegimeConfig':
        """Configuration optimized for pressure sensors."""
        return cls(
            derivative_threshold_low=1.5,
            derivative_threshold_high=3.5,
            volatility_threshold_low=0.7,
            volatility_threshold_high=2.5,
            min_regime_duration=300.0,
            use_derivative=True,
            use_rolling_std=True,
            use_second_derivative=True,
            use_lag_features=False,
        )
    
    @classmethod
    def for_vibration(cls) -> 'RegimeConfig':
        """Configuration optimized for vibration sensors."""
        return cls(
            derivative_threshold_low=2.0,
            derivative_threshold_high=4.0,
            volatility_threshold_low=1.0,
            volatility_threshold_high=3.0,
            min_regime_duration=180.0,  # 3 minutos (transiciones más rápidas)
            use_derivative=True,
            use_rolling_std=True,
            use_second_derivative=False,
            use_lag_features=True,
        )
