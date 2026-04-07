"""IoT domain entities.

DEPRECADO: Este módulo será eliminado en la migración completa a Zenin.
Usar: domain.entities.series + infrastructure.adapters.iot
"""

from __future__ import annotations

import warnings
warnings.warn(
    "domain.entities.iot está deprecado. "
    "Usar domain.entities.series + infrastructure.adapters.iot",
    DeprecationWarning,
    stacklevel=1,
)

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
