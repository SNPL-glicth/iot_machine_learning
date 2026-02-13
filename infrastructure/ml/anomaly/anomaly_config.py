"""Configuración para el ensemble de detección de anomalías.

Centraliza todos los parámetros que antes estaban hardcodeados
en voting_anomaly_detector.py y sus sub-detectores.

Sin I/O, sin sklearn — puro value object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class AnomalyDetectorConfig:
    """Configuración del ensemble de detección de anomalías.

    Attributes:
        contamination: Fracción esperada de anomalías (IF/LOF).
        voting_threshold: Score > threshold → anomalía.
        min_training_points: Mínimo de puntos para entrenar.
        n_estimators: Árboles en IsolationForest.
        random_state: Semilla para reproducibilidad.
        lof_max_neighbors: Máximo de vecinos para LOF.
        z_vote_lower: Z-score debajo del cual voto es 0.
        z_vote_upper: Z-score encima del cual voto es 1.
        weights: Pesos por método para voting ponderado.
    """

    contamination: float = 0.1
    voting_threshold: float = 0.5
    min_training_points: int = 50
    n_estimators: int = 100
    random_state: int = 42
    lof_max_neighbors: int = 20
    z_vote_lower: float = 2.0
    z_vote_upper: float = 3.0
    weights: Dict[str, float] = field(default_factory=lambda: {
        "isolation_forest": 0.30,
        "z_score": 0.20,
        "iqr": 0.10,
        "local_outlier_factor": 0.15,
        "velocity_z": 0.15,
        "acceleration_z": 0.10,
    })

    def __post_init__(self) -> None:
        if not 0.0 < self.contamination < 0.5:
            raise ValueError(
                f"contamination debe estar en (0, 0.5), recibido {self.contamination}"
            )
        if not 0.0 < self.voting_threshold < 1.0:
            raise ValueError(
                f"voting_threshold debe estar en (0, 1), recibido {self.voting_threshold}"
            )
