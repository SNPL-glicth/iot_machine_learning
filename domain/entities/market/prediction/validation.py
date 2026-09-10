"""Validación pura de los campos de una predicción (FASE 3).

Funciones sin estado y sin I/O: el ``__post_init__`` de ``Prediction``
las usa para rechazar proyecciones imposibles antes de cualquier
persistencia. Mensajes estables (los tests dependen de ellos).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .distribution import ReturnDistribution
    from .types import PredictionInterval


def validate_horizon(horizon_seconds: int) -> None:
    """Valida el horizonte (int estricto, > 0)."""
    if not isinstance(horizon_seconds, int):
        raise TypeError("horizon_seconds debe ser int")
    if horizon_seconds <= 0:
        raise ValueError(f"horizon_seconds debe ser > 0: {horizon_seconds}")


def validate_expected_return(expected_return: float) -> None:
    """Valida el retorno esperado (finito)."""
    if not math.isfinite(expected_return):
        raise ValueError(f"expected_return inválido: {expected_return!r}")


def validate_interval_contains(
    interval: PredictionInterval, expected_return: float
) -> None:
    """Valida que la media quede dentro del intervalo (coherencia)."""
    if not interval.contains(expected_return):
        raise ValueError(
            "intervalo incoherente: expected_return fuera del intervalo "
            f"{expected_return} ∉ [{interval.lower}, {interval.upper}]"
        )


def validate_distribution_coherence(
    distribution: ReturnDistribution, expected_return: float
) -> None:
    """La media declarada y la de la distribución no pueden divergir."""
    if not math.isclose(
        distribution.expected_return, expected_return, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(
            "distribución incoherente: expected_return "
            f"{expected_return!r} != distribution.expected_return "
            f"{distribution.expected_return!r}"
        )
