"""
DynamicFeaturePipeline for orchestrating dynamic feature computation.

Coordinates the computation of all dynamic features including derivatives,
rolling statistics, lag features, and cross-features.
"""

from typing import Optional, Dict, List
import time

from .models.dynamic_features import DynamicFeatures
from .models.feature_config import FeatureConfig
from .rolling_window_engine import RollingWindowEngine

from ml_service.features.dynamic.derivative_computer import DerivativeCalculator
from ml_service.features.dynamic.lag_feature_generator import LagFeatureGenerator
from ml_service.features.dynamic.cross_feature_generator import CrossFeatureGenerator


class DynamicFeaturePipeline:
    """Orchestrates dynamic feature computation."""
    
    def __init__(
        self,
        rolling_window_engine: RollingWindowEngine,
        derivative_computer: Optional[DerivativeCalculator] = None,
        lag_feature_generator: Optional[LagFeatureGenerator] = None,
        cross_feature_generator: Optional[CrossFeatureGenerator] = None,
    ):
        """
        Initialize dynamic feature pipeline.
        
        Args:
            rolling_window_engine: Engine for managing rolling windows
            derivative_computer: Calculator for derivatives (default: create new)
            lag_feature_generator: Generator for lag features (default: create new)
            cross_feature_generator: Generator for cross-features (default: create new)
        """
        self._rolling_engine = rolling_window_engine
        self._derivative_computer = derivative_computer or DerivativeCalculator()
        self._lag_generator = lag_feature_generator or LagFeatureGenerator()
        self._cross_generator = cross_feature_generator or CrossFeatureGenerator()
    
    def compute(
        self,
        sensor_id: int,
        sensor_type: str,
        values: List[float],
        timestamps: List[float],
        current_value: float,
        current_timestamp: Optional[float] = None,
        config: Optional[FeatureConfig] = None,
        value_store: Optional[Dict[int, float]] = None,
        correlations: Optional[Dict[int, float]] = None,
    ) -> Optional[DynamicFeatures]:
        """
        Compute all dynamic features according to configuration.
        
        Args:
            sensor_id: Sensor identifier
            sensor_type: Sensor type (e.g., "TEMPERATURE", "PRESSURE")
            values: List of recent values (most recent last)
            timestamps: List of corresponding timestamps
            current_value: Current sensor value
            current_timestamp: Current timestamp (default: current time)
            config: Feature configuration (default: default config)
            value_store: Dictionary of latest values from other sensors (for cross-features)
            correlations: Dictionary of correlations with other sensors (for cross-features)
        
        Returns:
            DynamicFeatures object or None if computation fails
        """
        if current_timestamp is None:
            current_timestamp = time.time()
        
        if config is None:
            config = FeatureConfig.default()
        
        # Add reading to rolling windows
        self._rolling_engine.add_reading(sensor_id, current_value, current_timestamp)
        
        # Initialize dynamic features
        features = DynamicFeatures(
            sensor_id=sensor_id,
            timestamp=current_timestamp,
        )
        
        # Compute derivatives
        if config.enable_derivatives:
            features.derivative = self._derivative_computer.compute_first_derivative(
                values, timestamps
            )
            features.second_derivative = self._derivative_computer.compute_second_derivative(
                values, timestamps
            )
        
        # Compute rolling statistics
        if config.enable_rolling_stats:
            rolling_stats = self._rolling_engine.compute_stats(
                sensor_id, config.rolling_windows
            )
            
            # Map stats to fields
            if 60 in rolling_stats:
                stats = rolling_stats[60]
                features.rolling_mean_1h = stats.mean
                features.rolling_std_1h = stats.std
                features.rolling_min_1h = stats.min
                features.rolling_max_1h = stats.max
            
            if 360 in rolling_stats:
                stats = rolling_stats[360]
                features.rolling_mean_6h = stats.mean
                features.rolling_std_6h = stats.std
                features.rolling_min_6h = stats.min
                features.rolling_max_6h = stats.max
            
            if 1440 in rolling_stats:
                stats = rolling_stats[1440]
                features.rolling_mean_24h = stats.mean
                features.rolling_std_24h = stats.std
                features.rolling_min_24h = stats.min
                features.rolling_max_24h = stats.max
            
            # Compute volatility (CV = std/mean)
            if 60 in rolling_stats and rolling_stats[60].mean != 0:
                features.volatility_1h = rolling_stats[60].std / rolling_stats[60].mean
            if 360 in rolling_stats and rolling_stats[360].mean != 0:
                features.volatility_6h = rolling_stats[360].std / rolling_stats[360].mean
        
        # Compute lag features
        if config.enable_lag_features:
            lags = self._lag_generator.compute_lags(values, config.lag_periods)
            features.lag_1 = lags.get(1)
            features.lag_6 = lags.get(6)
            features.lag_24 = lags.get(24)
        
        # Compute cross-features
        if config.enable_cross_features and value_store:
            features.cross_features = self._cross_generator.compute_cross_features(
                sensor_id=sensor_id,
                current_value=current_value,
                value_store=value_store,
                correlations=correlations,
            )
        
        return features
    
    def add_reading(
        self,
        sensor_id: int,
        value: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Add a reading to the rolling windows (without computing features).
        
        This is useful for pre-populating windows before feature computation.
        
        Args:
            sensor_id: Sensor identifier
            value: Sensor value
            timestamp: Unix timestamp (default: current time)
        """
        self._rolling_engine.add_reading(sensor_id, value, timestamp)
