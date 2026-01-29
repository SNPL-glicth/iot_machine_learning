"""ML Features Producer - Observable ML that produces continuous signals.

REFACTORIZADO 2026-01-29:
- Modelos extraídos a models/
- Servicios extraídos a services/
- Este archivo ahora es solo el orquestador (~100 líneas, antes 501)

Estructura:
- models/ml_features.py: MLFeatures dataclass
- services/feature_computer.py: FeatureComputer
- services/window_manager.py: WindowManager

Key Features Produced:
- baseline: Expected value learned by the model
- deviation: Current deviation from baseline
- z_score: Normalized deviation (standard deviations from mean)
- trend_slope: Direction and rate of change
- stability_score: How stable the sensor is (0-1)
- confidence: Model confidence (0-1)
- pattern_detected: Identified behavior pattern

ISO 27001: Features are aggregated metrics, no raw sensitive data.
"""

from __future__ import annotations

import time
import logging
from typing import Dict, Optional, List

from .models import MLFeatures
from .services import FeatureComputer, WindowManager

logger = logging.getLogger(__name__)


class MLFeaturesProducer:
    """Produces observable ML features for every sensor reading.
    
    This transforms ML from a "black box that fires events" to an
    observable system that ALWAYS produces features, not just on anomalies.
    """
    
    def __init__(self, max_window_size: int = 100, max_window_age_seconds: float = 300.0):
        self._window_manager = WindowManager(max_window_size, max_window_age_seconds)
        self._feature_computer = FeatureComputer()
    
    def add_reading(self, sensor_id: int, value: float, timestamp: Optional[float] = None) -> Optional[MLFeatures]:
        """Add a reading and compute features.
        
        Args:
            sensor_id: ID of the sensor
            value: Sensor reading value
            timestamp: Unix timestamp (default: current time)
            
        Returns:
            MLFeatures if enough data available, None otherwise
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Add to window
        self._window_manager.add_reading(sensor_id, value, timestamp)
        
        # Get window and compute features
        window = self._window_manager.get_window(sensor_id)
        if window is None or window.is_empty:
            return None
        
        return self._feature_computer.compute_features(window, value, timestamp)
    
    def get_sensor_features(self, sensor_id: int) -> Optional[MLFeatures]:
        """Get the latest features for a sensor (if available)."""
        window = self._window_manager.get_window(sensor_id)
        if window is None or window.is_empty:
            return None
        
        values = window.get_values()
        timestamps = window.get_timestamps()
        
        return self._feature_computer.compute_features(
            window,
            values[-1],
            timestamps[-1],
        )
    
    def get_all_sensor_ids(self) -> List[int]:
        """Get all sensor IDs being tracked."""
        return self._window_manager.get_all_sensor_ids()
    
    def get_statistics(self) -> Dict[int, Dict]:
        """Get statistics for all sensors."""
        return self._window_manager.get_statistics()
    
    def remove_sensor(self, sensor_id: int) -> None:
        """Remove a sensor from tracking."""
        self._window_manager.remove_sensor(sensor_id)
        logger.info(f"Removed sensor {sensor_id} from ML features tracking")
    
    def clear_all(self) -> None:
        """Clear all sensor data."""
        self._window_manager.clear_all()
        logger.info("Cleared all ML features data")


# Singleton instance
_producer_instance: Optional[MLFeaturesProducer] = None


def get_ml_features_producer() -> MLFeaturesProducer:
    """Get the singleton MLFeaturesProducer instance."""
    global _producer_instance
    if _producer_instance is None:
        _producer_instance = MLFeaturesProducer()
    return _producer_instance


def reset_ml_features_producer() -> None:
    """Reset the singleton instance (for testing)."""
    global _producer_instance
    if _producer_instance is not None:
        _producer_instance.clear_all()
    _producer_instance = None
