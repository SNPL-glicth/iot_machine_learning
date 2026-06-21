"""
Factory for creating and configuring regime detection components.

This module provides factory methods to create and wire together
all regime detection components with proper configuration.
"""

from typing import Optional, Dict

from .pipeline import RegimeDetectionPipeline
from .classifier import OperationalRegimeClassifier
from .state_manager import RegimeStateManager
from .metadata_registry import RegimeMetadataRegistry
from .router import ContextualAnomalyRouter
from .models.regime_config import RegimeConfig
from .models.anomaly_thresholds import AnomalyThresholds


class RegimeDetectionFactory:
    """Factory for creating regime detection components."""
    
    @staticmethod
    def create_default_pipeline(
        algorithm: str = "kmeans",
        n_components: int = 5,
        enable_cache: bool = True,
    ) -> RegimeDetectionPipeline:
        """
        Create a default RegimeDetectionPipeline with standard configuration.
        
        Args:
            algorithm: Algorithm to use ("kmeans", "gmm", "hdbscan", "hmm")
            n_components: Number of clusters/components
            enable_cache: Whether to enable caching in registry
        
        Returns:
            Configured RegimeDetectionPipeline
        """
        # Create classifier
        classifier = OperationalRegimeClassifier(
            algorithm=algorithm,
            n_components=n_components,
        )
        
        # Create state manager
        state_manager = RegimeStateManager(
            max_history_size=100,
            min_regime_duration=300.0,
        )
        
        # Create metadata registry with presets
        registry = RegimeDetectionFactory.create_registry_with_presets(enable_cache)
        
        # Create pipeline
        pipeline = RegimeDetectionPipeline(
            classifier=classifier,
            state_manager=state_manager,
            metadata_registry=registry,
        )
        
        return pipeline
    
    @staticmethod
    def create_minimal_pipeline() -> RegimeDetectionPipeline:
        """
        Create a minimal RegimeDetectionPipeline for quick wins.
        
        Uses K-Means with 5 clusters and heuristic fallback.
        
        Returns:
            Configured RegimeDetectionPipeline with minimal features
        """
        classifier = OperationalRegimeClassifier(
            algorithm="kmeans",
            n_components=5,
        )
        
        state_manager = RegimeStateManager(
            max_history_size=50,
            min_regime_duration=300.0,
        )
        
        registry = RegimeMetadataRegistry(enable_cache=True)
        
        pipeline = RegimeDetectionPipeline(
            classifier=classifier,
            state_manager=state_manager,
            metadata_registry=registry,
        )
        
        return pipeline
    
    @staticmethod
    def create_registry_with_presets(enable_cache: bool = True) -> RegimeMetadataRegistry:
        """
        Create a RegimeMetadataRegistry with sensor type presets.
        
        Returns:
            Configured RegimeMetadataRegistry with TEMPERATURE, PRESSURE, VIBRATION presets
        """
        registry = RegimeMetadataRegistry(enable_cache=enable_cache)
        
        # Register sensor type presets
        registry.register_type_config("TEMPERATURE", RegimeConfig.for_temperature())
        registry.register_type_config("PRESSURE", RegimeConfig.for_pressure())
        registry.register_type_config("VIBRATION", RegimeConfig.for_vibration())
        
        return registry
    
    @staticmethod
    def create_router_with_thresholds() -> ContextualAnomalyRouter:
        """
        Create a ContextualAnomalyRouter with regime-specific thresholds.
        
        Returns:
            Configured ContextualAnomalyRouter
        """
        thresholds = {
            "STABLE_NORMAL": AnomalyThresholds(
                normal_multiplier=1.0,
                peak_load_multiplier=0.7,
                transition_multiplier=1.2,
                startup_multiplier=0.8,
            ),
            "VOLATILE_PEAK": AnomalyThresholds(
                normal_multiplier=0.7,
                peak_load_multiplier=0.5,
                transition_multiplier=1.0,
                startup_multiplier=0.7,
            ),
            "STARTUP": AnomalyThresholds(
                normal_multiplier=0.8,
                peak_load_multiplier=0.6,
                transition_multiplier=1.0,
                startup_multiplier=0.5,
            ),
            "SHUTDOWN": AnomalyThresholds(
                normal_multiplier=0.8,
                peak_load_multiplier=0.6,
                transition_multiplier=1.0,
                startup_multiplier=0.5,
            ),
            "ANOMALOUS_REGIME": AnomalyThresholds(
                normal_multiplier=1.0,
                peak_load_multiplier=1.0,
                transition_multiplier=1.0,
                startup_multiplier=1.0,
            ),
        }
        
        return ContextualAnomalyRouter(regime_thresholds=thresholds)
