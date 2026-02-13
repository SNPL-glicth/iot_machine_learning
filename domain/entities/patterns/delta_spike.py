"""DeltaSpikeResult — clasificación de spike (delta vs noise).

Entidad pura del dominio — sin dependencias de infraestructura.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SpikeClassification(Enum):
    """Clasificación de un spike detectado."""

    DELTA_SPIKE = "delta_spike"
    NOISE_SPIKE = "noise_spike"
    NORMAL = "normal"


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
