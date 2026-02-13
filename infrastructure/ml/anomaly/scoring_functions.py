"""Funciones de scoring puras para detección de anomalías.

Sin estado, sin I/O, sin sklearn. Cada función transforma un valor
en un voto [0, 1] o calcula una métrica estadística.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple


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


def compute_z_vote(
    z_score: float,
    lower: float = 2.0,
    upper: float = 3.0,
) -> float:
    """Convierte Z-score en voto de anomalía [0, 1].

    Regla:
    - z > upper → 1.0 (anomalía clara)
    - lower < z <= upper → lineal entre 0 y 1
    - z <= lower → 0.0 (normal)

    Args:
        z_score: Z-score absoluto.
        lower: Z-score below which vote is 0.0.
        upper: Z-score above which vote is 1.0.

    Returns:
        Voto en [0.0, 1.0].
    """
    if z_score > upper:
        return 1.0
    if z_score > lower:
        span = upper - lower
        return (z_score - lower) / span if span > 0 else 1.0
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
