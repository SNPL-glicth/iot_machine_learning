"""Métricas distribucionales puras (FASE 1).

pinball (por cuantil), CRPS gaussiano (forma cerrada) y cobertura de
banda. Matemática sin estado ni I/O; la evaluación contra el Outcome
vive en ``evaluate_distribution``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .distribution import ReturnDistribution

__all__ = [
    "DistributionEvaluation",
    "pinball_loss",
    "crps_gaussian",
    "evaluate_distribution",
]

_INV_SQRT_PI: float = 1.0 / math.sqrt(math.pi)


def pinball_loss(level: float, qvalue: float, realized: float) -> float:
    """Pérdida pinball del cuantil τ: optima cuando q es el cuantil τ."""
    if not 0.0 < level < 1.0:
        raise ValueError(f"level fuera de (0, 1): {level!r}")
    if not math.isfinite(qvalue) or not math.isfinite(realized):
        raise ValueError(f"qvalue/realized no finitos: {qvalue!r}/{realized!r}")
    if realized >= qvalue:
        return (realized - qvalue) * level
    return (qvalue - realized) * (1.0 - level)


def crps_gaussian(mean: float, std: float, realized: float) -> float:
    """CRPS cerrado de N(mean, std) contra el realizado (>= 0)."""
    if not math.isfinite(mean) or not math.isfinite(realized):
        raise ValueError("mean/realized no finitos")
    if not math.isfinite(std) or std <= 0:
        raise ValueError(f"std debe ser > 0: {std!r}")
    z = (realized - mean) / std
    phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return std * (z * (2.0 * phi - 1.0) + 2.0 * pdf - _INV_SQRT_PI)


@dataclass(frozen=True, slots=True, kw_only=True)
class DistributionEvaluation:
    """Evaluación de la distribución contra el retorno realizado."""

    pinball_10: float
    pinball_50: float
    pinball_90: float
    mean_pinball: float
    crps: float
    within_band: bool
    tail_breach: bool  # realized < cvar_05 (False si no hay cvar)

    def __post_init__(self) -> None:
        for name in ("pinball_10", "pinball_50", "pinball_90",
                     "mean_pinball", "crps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} inválido: {value!r}")


def evaluate_distribution(
    distribution: ReturnDistribution, return_realized: float
) -> DistributionEvaluation:
    """Evalúa cuantiles + forma contra el retorno realizado."""
    if not math.isfinite(return_realized):
        raise ValueError(f"return_realized inválido: {return_realized!r}")
    p10 = pinball_loss(0.10, distribution.quantile_10, return_realized)
    p50 = pinball_loss(0.50, distribution.quantile_50, return_realized)
    p90 = pinball_loss(0.90, distribution.quantile_90, return_realized)
    cvar = distribution.cvar_05
    return DistributionEvaluation(
        pinball_10=p10,
        pinball_50=p50,
        pinball_90=p90,
        mean_pinball=(p10 + p50 + p90) / 3.0,
        crps=crps_gaussian(
            distribution.expected_return, distribution.volatility,
            return_realized,
        ),
        within_band=distribution.within_band(return_realized),
        tail_breach=bool(cvar is not None and return_realized < cvar),
    )
