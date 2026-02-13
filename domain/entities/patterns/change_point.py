"""ChangePoint — punto de cambio estructural en una serie temporal.

Entidad pura del dominio — sin dependencias de infraestructura.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ChangePointType(Enum):
    """Tipos de cambio estructural en la serie temporal."""

    LEVEL_SHIFT = "level_shift"
    TREND_CHANGE = "trend_change"
    VARIANCE_CHANGE = "variance_change"


@dataclass(frozen=True)
class ChangePoint:
    """Punto de cambio estructural detectado en la serie.

    Attributes:
        index: Índice en la serie temporal donde ocurre el cambio.
        timestamp: Timestamp Unix del cambio (si disponible).
        change_type: Tipo de cambio (nivel, tendencia, varianza).
        magnitude: Magnitud del cambio.
        confidence: Confianza en la detección (0.0–1.0).
        before_mean: Media antes del cambio.
        after_mean: Media después del cambio.
    """

    index: int
    timestamp: float = 0.0
    change_type: ChangePointType = ChangePointType.LEVEL_SHIFT
    magnitude: float = 0.0
    confidence: float = 0.0
    before_mean: Optional[float] = None
    after_mean: Optional[float] = None
