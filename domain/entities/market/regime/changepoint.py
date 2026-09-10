"""Detección de cambio de régimen — CUSUM secuencial (FASE 2).

Baseline con los primeros ``warmup`` retornos; el CUSUM corre solo
hacia adelante sobre lo posterior. Así un cambio sostenido dispara
cerca del cambio real y la mitad tranquila no dispara (centrar en la
media global haría ambas cosas indistinguibles).

Señal para el filtro y el loop de adaptación (Fase 7): un changepoint
cuestiona al MAP actual.
"""

from __future__ import annotations

import math

__all__ = ["cusum_changepoint"]


def cusum_changepoint(
    returns: tuple[float, ...] | list[float],
    *,
    warmup: int = 20,
    drift: float = 0.5,
    threshold: float = 5.0,
    min_returns: int = 10,
) -> int | None:
    """Índice global del primer cruce CUSUM (None si no hay cambio).

    Args:
        returns: Retornos en orden temporal (cerrados, sin futuro).
        warmup: Muestras iniciales para la baseline (media/σ).
        drift: Tolerancia en σ (ignora derivas menores).
        threshold: Umbral de suma acumulada en σ.
        min_returns: Mínimo de muestras totales.
    """
    returns = tuple(returns)
    n = len(returns)
    if n < min_returns:
        raise ValueError(
            f"retornos insuficientes: {n} < {min_returns}"
        )
    if warmup < 3 or warmup >= n:
        raise ValueError(f"warmup inválido: {warmup} para {n} retornos")
    if any(not math.isfinite(r) for r in returns):
        raise ValueError("retornos no finitos")
    if drift < 0 or threshold <= 0:
        raise ValueError("drift debe ser >= 0 y threshold > 0")

    base = returns[:warmup]
    mean = sum(base) / warmup
    var = sum((r - mean) ** 2 for r in base) / warmup
    sigma = math.sqrt(var) if var > 0 else 1e-12

    upper = 0.0
    lower = 0.0
    for i in range(warmup, n):
        z = (returns[i] - mean) / sigma
        upper = max(0.0, upper + z - drift)
        lower = min(0.0, lower + z + drift)
        if upper > threshold or lower < -threshold:
            return i
    return None
