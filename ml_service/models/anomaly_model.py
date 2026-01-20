from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnomalyModel:
    """Modelo de anomalía (Isolation Forest) por sensor.

    Solo guarda metadatos imprescindibles; el estimador de sklearn se mantiene en memoria
    dentro del trainer.
    """

    sensor_id: int
    # Umbral sobre score_samples para marcar anomalía.
    # Los scores típicamente son negativos; cuanto más negativo, más anómalo.
    threshold_score: float
    # Rango observado de scores durante el entrenamiento, usado para normalizar.
    score_min: float
    score_max: float
