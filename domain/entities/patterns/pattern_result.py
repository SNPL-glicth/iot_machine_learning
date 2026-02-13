"""PatternResult — resultado de detección de patrón de comportamiento.

Entidad pura del dominio — sin dependencias de infraestructura.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class PatternType(Enum):
    """Tipos de patrón de comportamiento detectables."""

    STABLE = "stable"
    DRIFTING = "drifting"
    OSCILLATING = "oscillating"
    SPIKE = "spike"
    MICRO_VARIATION = "micro_variation"
    CURVE_ANOMALY = "curve_anomaly"
    REGIME_TRANSITION = "regime_transition"


@dataclass(frozen=True)
class PatternResult:
    """Resultado de detección de patrón de comportamiento.

    Attributes:
        series_id: Identificador de la serie analizada.
        pattern_type: Tipo de patrón detectado.
        confidence: Confianza en la detección (0.0–1.0).
        description: Descripción legible del patrón.
        metadata: Información adicional (slopes, z-scores, etc.).
    """

    series_id: str
    pattern_type: PatternType
    confidence: float
    description: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)
