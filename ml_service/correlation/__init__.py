"""Módulo de correlación de sensores.

Modular implementation:
    - types: Entities and enums
    - queries: Database queries
    - pattern_matcher: Correlation logic
    - sensor_correlator: Main facade (backward compatibility)
"""

from .sensor_correlator import correlate_sensor_with_device
from .pattern_matcher import SensorCorrelator
from .types import (
    CorrelationPattern,
    CorrelationResult,
    DeviceSensorGroup,
    SensorSnapshot,
)

__all__ = [
    "CorrelationPattern",
    "SensorSnapshot",
    "CorrelationResult",
    "DeviceSensorGroup",
    "SensorCorrelator",
    "correlate_sensor_with_device",
]
