"""Módulo de correlación de sensores."""

from .sensor_correlator import (
    CorrelationPattern,
    SensorSnapshot,
    CorrelationResult,
    DeviceSensorGroup,
    SensorCorrelator,
    correlate_sensor_with_device,
)

__all__ = [
    "CorrelationPattern",
    "SensorSnapshot",
    "CorrelationResult",
    "DeviceSensorGroup",
    "SensorCorrelator",
    "correlate_sensor_with_device",
]
