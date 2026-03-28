"""Configuración para el ensemble de detección de anomalías.

Centraliza todos los parámetros que antes estaban hardcodeados
en voting_anomaly_detector.py, change_point_detector.py y delta_spike_classifier.py.

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
        
        # CUSUM change point detection
        cusum_threshold: Umbral acumulador para detectar cambio (menor = más sensible).
        cusum_drift: Mínimo cambio a detectar (slack parameter).
        
        # Delta spike classification
        delta_magnitude_sigma: Umbral en sigmas para considerar spike.
        delta_persistence_window: Lecturas post-spike para evaluar persistencia.
        delta_min_history: Mínimo de puntos pre-spike para estadísticas.
        delta_persistence_score_threshold: Score mínimo para considerar persistente.
        delta_trend_alignment_threshold: Umbral de alineación con tendencia.
    """

    # Anomaly detection ensemble
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
    
    # CUSUM change point detection (moved from hardcoded defaults)
    cusum_threshold: float = 5.0
    cusum_drift: float = 0.5
    
    # Delta spike classification (moved from hardcoded defaults)
    delta_magnitude_sigma: float = 2.0
    delta_persistence_window: int = 5
    delta_min_history: int = 20
    delta_persistence_score_threshold: float = 0.6
    delta_trend_alignment_threshold: float = 0.8

    def __post_init__(self) -> None:
        # Validate anomaly detection params
        if not 0.0 < self.contamination < 0.5:
            raise ValueError(
                f"contamination debe estar en (0, 0.5), recibido {self.contamination}"
            )
        if not 0.0 < self.voting_threshold < 1.0:
            raise ValueError(
                f"voting_threshold debe estar en (0, 1), recibido {self.voting_threshold}"
            )
        
        # Validate weights sum to 1.0 (with small tolerance for float errors)
        total_weight = sum(self.weights.values())
        if not 0.99 <= total_weight <= 1.01:
            raise ValueError(
                f"weights deben sumar 1.0, suman {total_weight}"
            )
        
        # Validate CUSUM params
        if self.cusum_threshold <= 0:
            raise ValueError(
                f"cusum_threshold debe ser > 0, recibido {self.cusum_threshold}"
            )
        if self.cusum_drift < 0:
            raise ValueError(
                f"cusum_drift debe ser >= 0, recibido {self.cusum_drift}"
            )
        
        # Validate delta spike params
        if self.delta_magnitude_sigma <= 0:
            raise ValueError(
                f"delta_magnitude_sigma debe ser > 0, recibido {self.delta_magnitude_sigma}"
            )
        if self.delta_persistence_window < 2:
            raise ValueError(
                f"delta_persistence_window debe ser >= 2, recibido {self.delta_persistence_window}"
            )
        if self.delta_min_history < 5:
            raise ValueError(
                f"delta_min_history debe ser >= 5, recibido {self.delta_min_history}"
            )
