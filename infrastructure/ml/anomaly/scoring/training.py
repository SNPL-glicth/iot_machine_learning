"""Estadísticas de entrenamiento para detección de anomalías.

Value objects puros + función de cómputo. Sin I/O, sin sklearn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class TrainingStats:
    """Estadísticas calculadas durante entrenamiento.

    Value object puro — sin lógica de negocio.
    """

    mean: float
    std: float
    q1: float
    q3: float
    iqr: float


def compute_training_stats(values: List[float]) -> TrainingStats:
    """Calcula estadísticas de entrenamiento a partir de datos históricos.

    Args:
        values: Serie temporal (>= 1 punto).

    Returns:
        ``TrainingStats`` con mean, std, Q1, Q3, IQR.
    """
    n = len(values)
    if n == 0:
        return TrainingStats(mean=0.0, std=1e-9, q1=0.0, q3=0.0, iqr=0.0)

    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    if std < 1e-9:
        std = 1e-9

    sorted_vals = sorted(values)
    q1_idx = int(n * 0.25)
    q3_idx = int(n * 0.75)
    q1 = sorted_vals[q1_idx]
    q3 = sorted_vals[q3_idx]
    iqr = q3 - q1

    return TrainingStats(mean=mean, std=std, q1=q1, q3=q3, iqr=iqr)
