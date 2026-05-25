"""Configuración del ensemble de detección de anomalías.
Value object puro. Sin lógica, sin I/O, sin imports de detectores.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class AnomalyDetectorConfig:

    contamination: float = 0.005
    voting_threshold: float = 0.75
    min_training_points: int = 50
    n_estimators: int = 100
    random_state: int = 42
    lof_max_neighbors: int = 20
    z_vote_lower: float = 2.5
    z_vote_upper: float = 3.0

    weights: Dict[str, float] = field(default_factory=lambda: {
        "isolation_forest": 0.25,
        "z_score":          0.20,
        "rolling_z":        0.20,
        "velocity_z":       0.15,
        "acceleration_z":   0.10,
        "iqr":              0.05,
        "local_outlier_factor": 0.05,
    })

    def __post_init__(self) -> None:
        if not 0.0 < self.contamination < 0.5:
            raise ValueError(f"contamination must be in (0, 0.5)")
        if not 0.0 < self.voting_threshold < 1.0:
            raise ValueError(f"voting_threshold must be in (0, 1)")
        total = sum(self.weights.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"weights must sum to 1.0, got {total}")
