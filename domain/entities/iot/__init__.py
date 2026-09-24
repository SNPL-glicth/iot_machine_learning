"""IoT domain entities — legacy boundary.

SensorReading y SensorWindow son el punto de entrada IoT al sistema.
Para series agnósticas, usar ``series.TimeSeries``.
"""

from __future__ import annotations

from .sensor_profile import SensorProfile
from .sensor_ranges import DEFAULT_SENSOR_RANGES, get_default_range
from .sensor_reading import Reading, SensorReading, SensorWindow

__all__ = [
    "SensorReading",
    "SensorWindow",
    "Reading",
    "DEFAULT_SENSOR_RANGES",
    "get_default_range",
    "SensorProfile",
]

