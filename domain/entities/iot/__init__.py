"""Entidades específicas de IoT — legacy boundary.

SensorReading y SensorWindow son el punto de entrada IoT al sistema.
Para series agnósticas, usar ``series.TimeSeries``.
"""

from .sensor_reading import SensorReading, SensorWindow
from .sensor_ranges import DEFAULT_SENSOR_RANGES, get_default_range

__all__ = [
    "SensorReading",
    "SensorWindow",
    "DEFAULT_SENSOR_RANGES",
    "get_default_range",
]
