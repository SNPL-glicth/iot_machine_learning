"""Configuración del ensemble de detección de anomalías.
Value object puro. Sin lógica, sin I/O, sin imports de detectores.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AnomalyDetectorConfig:

    contamination: float = 0.005
    voting_threshold: float = 0.65
    min_training_points: int = 50
    n_estimators: int = 100
    random_state: int = 42
    lof_max_neighbors: int = 20
    z_vote_lower: float = 2.5
    z_vote_upper: float = 3.0

    weights: dict[str, float] = field(default_factory=lambda: {
        "z_score":              0.30,
        "isolation_forest":     0.25,
        "cumulative_residual":  0.25,
        "iqr":                  0.10,
        "velocity_z":           0.05,
        "rolling_z":            0.05,
        "acceleration_z":       0.00,
        "local_outlier_factor": 0.00,
    })

    def __post_init__(self) -> None:
        if not 0.0 < self.contamination < 0.5:
            raise ValueError("contamination must be in (0, 0.5)")
        if not 0.0 < self.voting_threshold < 1.0:
            raise ValueError("voting_threshold must be in (0, 1)")
        total = sum(self.weights.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"weights must sum to 1.0, got {total}")
