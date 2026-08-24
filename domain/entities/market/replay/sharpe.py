"""Sharpe ratio utility."""

from __future__ import annotations

from collections.abc import Sequence


def sharpe(values: Sequence[float]) -> float:
    """Sharpe de la lista de retornos (media / desviación muestral)."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    if var <= 0.0:
        return 0.0
    std = var**0.5
    return float(mean / std)