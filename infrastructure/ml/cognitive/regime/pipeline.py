"""
RegimeDetectionPipeline for orchestrating regime detection.

Coordinates OperationalRegimeClassifier and RegimeStateManager.
"""

import time
from typing import Optional

from .classifier import OperationalRegimeClassifier
from .state_manager import RegimeStateManager
from .metadata_registry import RegimeMetadataRegistry
from .models.regime_classification import RegimeClassification
from .models.regime_config import RegimeConfig


class RegimeDetectionPipeline:
    """Pipeline for detecting operational regimes."""
    
    def __init__(
        self,
        classifier: OperationalRegimeClassifier,
        state_manager: RegimeStateManager,
        metadata_registry: RegimeMetadataRegistry,
    ):
        """
        Initialize regime detection pipeline.
        
        Args:
            classifier: Operational regime classifier
            state_manager: Regime state manager
            metadata_registry: Regime metadata registry
        """
        self._classifier = classifier
        self._state_manager = state_manager
        self._registry = metadata_registry
    
    def detect_regime(
        self,
        sensor_id: int,
        sensor_type: str,
        dynamic_features,
        current_value: float,
        current_timestamp: Optional[float] = None,
    ) -> RegimeClassification:
        """
        Detect operational regime for a sensor.
        
        Args:
            sensor_id: Sensor identifier
            sensor_type: Sensor type (e.g., "TEMPERATURE", "PRESSURE")
            dynamic_features: Dynamic features from DynamicFeaturePipeline
            current_value: Current sensor value
            current_timestamp: Current timestamp (default: current time)
        
        Returns:
            Regime classification with context
        """
        if current_timestamp is None:
            current_timestamp = time.time()
        
        # Get configuration
        config = self._registry.get_config(sensor_id, sensor_type)
        
        # Classify regime
        regime_prediction = self._classifier.classify(
            dynamic_features=dynamic_features,
            config=config,
            current_value=current_value,
        )
        
        # Smooth transition with state manager
        smoothed_regime = self._state_manager.smooth_transition(
            sensor_id=sensor_id,
            new_regime=regime_prediction.regime,
            current_timestamp=current_timestamp,
            min_duration=config.min_regime_duration,
        )
        
        return RegimeClassification(
            sensor_id=sensor_id,
            timestamp=current_timestamp,
            regime=smoothed_regime,
            confidence=regime_prediction.confidence,
            previous_regime=self._state_manager.get_previous_regime(sensor_id),
            transition_duration=self._state_manager.get_transition_duration(sensor_id),
        )
