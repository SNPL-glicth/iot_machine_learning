"""IoT domain entities — legacy boundary.

SensorReading y SensorWindow son el punto de entrada IoT al sistema.
Para series agnósticas, usar ``series.TimeSeries``.
"""

from __future__ import annotations

from .sensor_reading import SensorReading, SensorWindow, Reading
from .sensor_ranges import DEFAULT_SENSOR_RANGES, get_default_range

__all__ = [
    "SensorReading",
    "SensorWindow",
    "Reading",
    "DEFAULT_SENSOR_RANGES",
    "get_default_range",
]
