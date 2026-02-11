"""Value Objects para detección de patrones, change points y regímenes.

Entidades puras del dominio — sin dependencias de infraestructura.
Cubren: patrones de comportamiento, puntos de cambio, clasificación
de spikes y regímenes operacionales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Optional


class PatternType(Enum):
    """Tipos de patrón de comportamiento detectables."""

    STABLE = "stable"
    DRIFTING = "drifting"
    OSCILLATING = "oscillating"
    SPIKE = "spike"
    MICRO_VARIATION = "micro_variation"
    CURVE_ANOMALY = "curve_anomaly"
    REGIME_TRANSITION = "regime_transition"


class ChangePointType(Enum):
    """Tipos de cambio estructural en la serie temporal."""

    LEVEL_SHIFT = "level_shift"        # Cambio abrupto de nivel
    TREND_CHANGE = "trend_change"      # Cambio en la pendiente
    VARIANCE_CHANGE = "variance_change"  # Cambio en la volatilidad


class SpikeClassification(Enum):
    """Clasificación de un spike detectado."""

    DELTA_SPIKE = "delta_spike"  # Cambio legítimo (válvula, encendido, etc.)
    NOISE_SPIKE = "noise_spike"  # Ruido / error de sensor
    NORMAL = "normal"            # No es spike


@dataclass(frozen=True)
class PatternResult:
    """Resultado de detección de patrón de comportamiento.

    Attributes:
        sensor_id: ID del sensor analizado.
        pattern_type: Tipo de patrón detectado.
        confidence: Confianza en la detección (0.0–1.0).
        description: Descripción legible del patrón.
        metadata: Información adicional (slopes, z-scores, etc.).
    """

    sensor_id: int
    pattern_type: PatternType
    confidence: float
    description: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ChangePoint:
    """Punto de cambio estructural detectado en la serie.

    Attributes:
        index: Índice en la serie temporal donde ocurre el cambio.
        timestamp: Timestamp Unix del cambio (si disponible).
        change_type: Tipo de cambio (nivel, tendencia, varianza).
        magnitude: Magnitud del cambio (unidades del sensor).
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


@dataclass(frozen=True)
class DeltaSpikeResult:
    """Resultado de clasificación de spike (delta vs noise).

    Attributes:
        is_delta_spike: ``True`` si es un cambio legítimo.
        confidence: Confianza en la clasificación (0.0–1.0).
        delta_magnitude: Magnitud absoluta del cambio.
        persistence_score: Qué tan persistente es el nuevo nivel (0–1).
        classification: Clasificación final.
        explanation: Explicación legible con razones cuantificadas.
        trend_alignment: Alineación con tendencia previa (0–1).
    """

    is_delta_spike: bool
    confidence: float
    delta_magnitude: float
    persistence_score: float
    classification: SpikeClassification
    explanation: str
    trend_alignment: float = 0.5


@dataclass(frozen=True)
class OperationalRegime:
    """Régimen operacional de un sensor.

    Representa un "modo" de operación (idle, activo, pico, enfriamiento).

    Attributes:
        regime_id: Identificador numérico del régimen.
        name: Nombre legible (``"idle"``, ``"active"``, ``"peak"``).
        mean_value: Valor medio típico en este régimen.
        std_value: Desviación estándar típica.
        typical_duration_s: Duración promedio en segundos.
    """

    regime_id: int
    name: str
    mean_value: float
    std_value: float
    typical_duration_s: float = 0.0
