"""Estadísticas temporales de entrenamiento para detección de anomalías.

Captura distribuciones de velocidad y aceleración para detectar
anomalías en la dinámica temporal de la serie.

Sin I/O, sin sklearn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class TemporalTrainingStats:
    """Estadísticas temporales calculadas durante entrenamiento.

    Attributes:
        vel_mean: Media de velocidades (dv/dt).
        vel_std: Desviación estándar de velocidades.
        vel_q1: Primer cuartil de velocidades.
        vel_q3: Tercer cuartil de velocidades.
        vel_iqr: Rango intercuartílico de velocidades.
        acc_mean: Media de aceleraciones (d²v/dt²).
        acc_std: Desviación estándar de aceleraciones.
        acc_q1: Primer cuartil de aceleraciones.
        acc_q3: Tercer cuartil de aceleraciones.
        acc_iqr: Rango intercuartílico de aceleraciones.
        has_temporal: True si se calcularon features temporales.
    """

    vel_mean: float = 0.0
    vel_std: float = 1e-9
    vel_q1: float = 0.0
    vel_q3: float = 0.0
    vel_iqr: float = 0.0
    acc_mean: float = 0.0
    acc_std: float = 1e-9
    acc_q1: float = 0.0
    acc_q3: float = 0.0
    acc_iqr: float = 0.0
    has_temporal: bool = False

    @classmethod
    def empty(cls) -> "TemporalTrainingStats":
        """Factory para cuando no hay datos temporales."""
        return cls()


def compute_temporal_training_stats(
    values: List[float],
    timestamps: List[float],
) -> TemporalTrainingStats:
    """Calcula estadísticas temporales de entrenamiento.

    Args:
        values: Valores históricos ordenados cronológicamente.
        timestamps: Timestamps correspondientes.

    Returns:
        ``TemporalTrainingStats`` con distribuciones de velocidad y aceleración.
    """
    from ....domain.validators.temporal_features import compute_temporal_features

    if len(values) < 3 or len(timestamps) < 3:
        return TemporalTrainingStats.empty()

    tf = compute_temporal_features(values, timestamps)

    if not tf.has_velocity:
        return TemporalTrainingStats.empty()

    vel_stats = _compute_distribution_stats(tf.velocities)
    acc_stats = _compute_distribution_stats(tf.accelerations) if tf.has_acceleration else _empty_dist()

    return TemporalTrainingStats(
        vel_mean=vel_stats[0],
        vel_std=vel_stats[1],
        vel_q1=vel_stats[2],
        vel_q3=vel_stats[3],
        vel_iqr=vel_stats[4],
        acc_mean=acc_stats[0],
        acc_std=acc_stats[1],
        acc_q1=acc_stats[2],
        acc_q3=acc_stats[3],
        acc_iqr=acc_stats[4],
        has_temporal=True,
    )


def _compute_distribution_stats(
    values: List[float],
) -> Tuple[float, float, float, float, float]:
    """Calcula (mean, std, q1, q3, iqr) de una lista de valores."""
    n = len(values)
    if n == 0:
        return _empty_dist()

    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / max(n - 1, 1))
    if std < 1e-9:
        std = 1e-9

    sorted_vals = sorted(values)
    q1 = sorted_vals[int(n * 0.25)]
    q3 = sorted_vals[int(n * 0.75)]
    iqr = q3 - q1

    return mean, std, q1, q3, iqr


def _empty_dist() -> Tuple[float, float, float, float, float]:
    """Distribución vacía por defecto."""
    return 0.0, 1e-9, 0.0, 0.0, 0.0
