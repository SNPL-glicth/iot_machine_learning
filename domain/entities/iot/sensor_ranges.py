"""Rangos operativos por tipo de sensor — LEGACY IoT.

.. deprecated::
    Usar ``SeriesContext.threshold`` (``series_context.py``) para rangos
    agnósticos al dominio.  Este módulo se mantiene solo por backward
    compatibility con ``ml_service/`` y ``severity_classifier.py``.

Responsabilidad ÚNICA: definir rangos físicos recomendados por tipo de sensor IoT.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

# Rangos operativos recomendados por tipo de sensor.
# Fuente: estándares industriales y buenas prácticas de mantenimiento.
# Formato: sensor_type → (min_safe, max_safe)
DEFAULT_SENSOR_RANGES: Dict[str, Tuple[float, float]] = {
    "temperature": (15.0, 35.0),
    "humidity": (30.0, 70.0),
    "air_quality": (400.0, 1000.0),
    "power": (0.0, 100000.0),
    "voltage": (0.0, 100000.0),
}


def get_default_range(sensor_type: str) -> Optional[Tuple[float, float]]:
    """Obtiene el rango operativo recomendado para un tipo de sensor.

    .. deprecated::
        Usar ``classify_severity_agnostic(value, threshold=...)`` en su lugar.

    Args:
        sensor_type: Tipo de sensor (e.g. "temperature", "humidity").

    Returns:
        Tupla ``(min_safe, max_safe)`` o ``None`` si no hay rango definido.
    """
    warnings.warn(
        "get_default_range(sensor_type) is deprecated. "
        "Use classify_severity_agnostic(value, threshold=...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return DEFAULT_SENSOR_RANGES.get(sensor_type)
