"""
Factory for creating and configuring dynamic feature components.

This module provides factory methods to create and wire together
all dynamic feature components with proper configuration.
"""

from typing import Optional

from infrastructure.ml.cognitive.dynamic.pipeline import DynamicFeaturePipeline
from infrastructure.ml.cognitive.dynamic.rolling_window_engine import RollingWindowEngine
from infrastructure.ml.cognitive.dynamic.feature_metadata_registry import FeatureMetadataRegistry
from infrastructure.ml.cognitive.dynamic.models.feature_config import FeatureConfig

from ml_service.features.dynamic.derivative_computer import DerivativeCalculator
from ml_service.features.dynamic.lag_feature_generator import LagFeatureGenerator
from ml_service.features.dynamic.cross_feature_generator import CrossFeatureGenerator


class DynamicFeatureFactory:
    """Factory for creating dynamic feature components."""
    
    @staticmethod
    def create_default_pipeline(
        window_sizes_minutes: Optional[list] = None,
        enable_persistence: bool = False,
    ) -> DynamicFeaturePipeline:
        """
        Create a default DynamicFeaturePipeline with standard configuration.
        
        Args:
            window_sizes_minutes: Rolling window sizes in minutes (default: [60, 360, 1440])
            enable_persistence: Whether to enable persistence (default: False)
        
        Returns:
            Configured DynamicFeaturePipeline
        """
        window_sizes = window_sizes_minutes or [60, 360, 1440]
        
        # Create rolling window engine
        rolling_engine = RollingWindowEngine(
            window_sizes_minutes=window_sizes,
            enable_persistence=enable_persistence,
        )
        
        # Create feature calculators
        derivative_computer = DerivativeCalculator(
            smoothing_window=5,
            max_gap_seconds=300.0,
        )
        
        lag_generator = LagFeatureGenerator(
            default_lag_periods=[1, 6, 24],
        )
        
        cross_generator = CrossFeatureGenerator(
            correlation_threshold=0.7,
            max_cross_features=5,
        )
        
        # Create pipeline
        pipeline = DynamicFeaturePipeline(
            rolling_window_engine=rolling_engine,
            derivative_computer=derivative_computer,
            lag_feature_generator=lag_generator,
            cross_feature_generator=cross_generator,
        )
        
        return pipeline
    
    @staticmethod
    def create_minimal_pipeline() -> DynamicFeaturePipeline:
        """
        Create a minimal DynamicFeaturePipeline for quick wins.
        
        Only computes derivatives and 1h rolling stats.
        
        Returns:
            Configured DynamicFeaturePipeline with minimal features
        """
        rolling_engine = RollingWindowEngine(
            window_sizes_minutes=[60],  # Only 1h
            enable_persistence=False,
        )
        
        derivative_computer = DerivativeCalculator(
            smoothing_window=5,
            max_gap_seconds=300.0,
        )
        
        pipeline = DynamicFeaturePipeline(
            rolling_window_engine=rolling_engine,
            derivative_computer=derivative_computer,
            lag_feature_generator=None,  # Disabled
            cross_feature_generator=None,  # Disabled
        )
        
        return pipeline
    
    @staticmethod
    def create_registry_with_presets() -> FeatureMetadataRegistry:
        """
        Create a FeatureMetadataRegistry with sensor type presets.
        
        Returns:
            Configured FeatureMetadataRegistry with TEMPERATURE, PRESSURE, VIBRATION presets
        """
        registry = FeatureMetadataRegistry(enable_cache=True)
        
        # Register sensor type presets
        registry.register_type_config("TEMPERATURE", FeatureConfig.for_temperature())
        registry.register_type_config("PRESSURE", FeatureConfig.for_pressure())
        registry.register_type_config("VIBRATION", FeatureConfig.for_vibration())
        
        return registry
    
    @staticmethod
    def create_for_sensor_type(sensor_type: str) -> tuple[DynamicFeaturePipeline, FeatureConfig]:
        """
        Create a pipeline and config optimized for a specific sensor type.
        
        Args:
            sensor_type: Sensor type (e.g., "TEMPERATURE", "PRESSURE", "VIBRATION")
        
        Returns:
            Tuple of (DynamicFeaturePipeline, FeatureConfig)
        """
        # Get preset config for sensor type
        if sensor_type == "TEMPERATURE":
            config = FeatureConfig.for_temperature()
            window_sizes = [60, 360, 1440]
        elif sensor_type == "PRESSURE":
            config = FeatureConfig.for_pressure()
            window_sizes = [60, 360]
        elif sensor_type == "VIBRATION":
            config = FeatureConfig.for_vibration()
            window_sizes = [30, 60]
        else:
            config = FeatureConfig.default()
            window_sizes = [60, 360, 1440]
        
        # Create pipeline with appropriate window sizes
        pipeline = DynamicFeatureFactory.create_default_pipeline(
            window_sizes_minutes=window_sizes,
        )
        
        return pipeline, config
