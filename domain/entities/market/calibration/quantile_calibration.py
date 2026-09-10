"""Calibración por cuantil (FASE 1).

Corrección conformal aditiva por nivel: sobre una ventana de
calibración (sin leakage: pasado ya realizado) se mide el sesgo del
cuantil declarado y se desplaza para recuperar cobertura nominal.

    shift_τ = cuantil_τ(realizado - q_declarado)

El shift se estima por (estrategia × horizonte × régimen) igual que la
calibración de probabilidades; aquí vive solo la matemática pura.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..prediction.distribution import QUANTILE_LEVELS, ReturnDistribution

__all__ = [
    "QuantileShift",
    "fit_quantile_shift",
    "fit_quantile_shifts",
    "apply_quantile_shifts",
    "empirical_coverage",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class QuantileShift:
    """Corrección aditiva de un cuantil, con su evidencia."""

    level: float
    shift: float
    samples: int
    coverage_before: float

    def __post_init__(self) -> None:
        if not 0.0 < self.level < 1.0:
            raise ValueError(f"level fuera de (0, 1): {self.level!r}")
        if not math.isfinite(self.shift):
            raise ValueError(f"shift inválido: {self.shift!r}")
        if self.samples < 0:
            raise ValueError("samples debe ser >= 0")
        if not 0.0 <= self.coverage_before <= 1.0:
            raise ValueError(
                f"coverage_before fuera de [0, 1]: {self.coverage_before!r}"
            )


def empirical_coverage(qvalues: list[float], realizeds: list[float]) -> float:
    """Fracción de realizados <= cuantil declarado (debería ≈ τ)."""
    if len(qvalues) != len(realizeds):
        raise ValueError("qvalues y realizeds deben tener igual longitud")
    if not qvalues:
        return 0.0
    hits = sum(1 for q, y in zip(qvalues, realizeds) if y <= q)
    return hits / len(qvalues)


def _residual_quantile(residuals: list[float], level: float) -> float:
    """Cuantil empírico (interpolación lineal, determinista)."""
    ordered = sorted(residuals)
    pos = level * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def fit_quantile_shift(
    level: float,
    qvalues: list[float],
    realizeds: list[float],
    *,
    min_samples: int = 3,
) -> QuantileShift:
    """Estima el shift aditivo para un nivel (0.0 sin evidencia)."""
    if not 0.0 < level < 1.0:
        raise ValueError(f"level fuera de (0, 1): {level!r}")
    if len(qvalues) != len(realizeds):
        raise ValueError("qvalues y realizeds deben tener igual longitud")
    coverage = empirical_coverage(qvalues, realizeds)
    if len(qvalues) < min_samples:
        return QuantileShift(
            level=level, shift=0.0, samples=len(qvalues),
            coverage_before=coverage,
        )
    residuals = [y - q for q, y in zip(qvalues, realizeds)]
    return QuantileShift(
        level=level,
        shift=_residual_quantile(residuals, level),
        samples=len(qvalues),
        coverage_before=coverage,
    )


def fit_quantile_shifts(
    qvalues_by_level: dict[float, list[float]],
    realizeds: list[float],
    *,
    min_samples: int = 3,
) -> dict[float, QuantileShift]:
    """Ajusta los niveles estándar q10/q50/q90 de una tacada."""
    fitted: dict[float, QuantileShift] = {}
    for level in QUANTILE_LEVELS:
        fitted[level] = fit_quantile_shift(
            level, qvalues_by_level.get(level, []), realizeds,
            min_samples=min_samples,
        )
    return fitted


def apply_quantile_shifts(
    distribution: ReturnDistribution,
    shifts: dict[float, QuantileShift | float],
) -> ReturnDistribution:
    """Retorna la distribución con cuantiles desplazados (inmutable)."""

    def _value(level: float) -> float:
        item = shifts.get(level, 0.0)
        return item.shift if isinstance(item, QuantileShift) else float(item)

    return replace(
        distribution,
        quantile_10=distribution.quantile_10 + _value(0.10),
        quantile_50=distribution.quantile_50 + _value(0.50),
        quantile_90=distribution.quantile_90 + _value(0.90),
    )
