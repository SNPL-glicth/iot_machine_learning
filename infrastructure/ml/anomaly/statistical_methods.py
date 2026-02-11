"""Métodos estadísticos puros para detección de anomalías.

Extraído de voting_anomaly_detector.py.
Responsabilidad ÚNICA: cálculos matemáticos sin estado, sin I/O, sin sklearn.

Funciones:
- compute_z_score: Desviación estándar normalizada.
- compute_z_vote: Voto de anomalía basado en Z-score.
- compute_iqr_bounds: Límites IQR (Q1 - 1.5·IQR, Q3 + 1.5·IQR).
- compute_iqr_vote: Voto de anomalía basado en IQR.
- weighted_vote: Promedio ponderado de votos.
- compute_consensus_confidence: Confianza basada en varianza de votos.
- compute_training_stats: Estadísticas de entrenamiento (mean, std, Q1, Q3, IQR).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple


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


def compute_z_score(value: float, mean: float, std: float) -> float:
    """Calcula Z-score absoluto.

    Args:
        value: Valor a evaluar.
        mean: Media de la distribución.
        std: Desviación estándar (> 0).

    Returns:
        Z-score absoluto (>= 0).
    """
    if std < 1e-12:
        return 0.0
    return abs(value - mean) / std


def compute_z_vote(z_score: float) -> float:
    """Convierte Z-score en voto de anomalía [0, 1].

    Regla:
    - z > 3.0 → 1.0 (anomalía clara)
    - 2.0 < z <= 3.0 → lineal entre 0 y 1
    - z <= 2.0 → 0.0 (normal)

    Args:
        z_score: Z-score absoluto.

    Returns:
        Voto en [0.0, 1.0].
    """
    if z_score > 3.0:
        return 1.0
    if z_score > 2.0:
        return (z_score - 2.0) / 1.0
    return 0.0


def compute_iqr_bounds(
    q1: float, q3: float, iqr: float
) -> Tuple[float, float]:
    """Calcula límites IQR (Tukey fences).

    Args:
        q1: Primer cuartil.
        q3: Tercer cuartil.
        iqr: Rango intercuartílico (Q3 - Q1).

    Returns:
        Tupla ``(lower_bound, upper_bound)``.
    """
    return (q1 - 1.5 * iqr, q3 + 1.5 * iqr)


def compute_iqr_vote(
    value: float, q1: float, q3: float, iqr: float
) -> float:
    """Voto de anomalía basado en IQR.

    Args:
        value: Valor a evaluar.
        q1: Primer cuartil.
        q3: Tercer cuartil.
        iqr: Rango intercuartílico.

    Returns:
        1.0 si fuera de rango IQR, 0.0 si dentro.
    """
    if iqr < 1e-9:
        return 0.0
    lower, upper = compute_iqr_bounds(q1, q3, iqr)
    return 1.0 if (value < lower or value > upper) else 0.0


def weighted_vote(
    votes: Dict[str, float],
    weights: Dict[str, float],
    default_weight: float = 0.1,
) -> float:
    """Calcula promedio ponderado de votos.

    Args:
        votes: Dict método → voto [0, 1].
        weights: Dict método → peso.
        default_weight: Peso para métodos sin peso definido.

    Returns:
        Score final en [0, 1].
    """
    total_weight = 0.0
    weighted_sum = 0.0

    for method, vote in votes.items():
        w = weights.get(method, default_weight)
        weighted_sum += vote * w
        total_weight += w

    return weighted_sum / total_weight if total_weight > 0 else 0.0


def compute_consensus_confidence(votes: Dict[str, float]) -> float:
    """Calcula confianza basada en consenso entre votantes.

    Mayor consenso (menor varianza) → mayor confianza.

    Args:
        votes: Dict método → voto [0, 1].

    Returns:
        Confianza en [0.5, 1.0].
    """
    if len(votes) <= 1:
        return 0.6

    vote_values = list(votes.values())
    vote_mean = sum(vote_values) / len(vote_values)
    vote_var = sum((v - vote_mean) ** 2 for v in vote_values) / len(vote_values)

    return max(0.5, 1.0 - math.sqrt(vote_var))
