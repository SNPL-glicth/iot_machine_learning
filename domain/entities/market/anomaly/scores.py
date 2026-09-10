"""Scores robustos de anomalía (FASE 4).

Hampel-style: mediana ± MAD (×0.6745⁻¹ para escala σ). Robusto a las
propias anomalías que busca — la media/desvío clásicos se los tragarían.

Cada detector mira el ÚLTIMO punto contra su baseline (pasado cerrado,
sin futuro) y devuelve AnomalyScore con breach explícito. Lo que cada
score significa y sus umbrales viven aquí, no dispersos en callers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "AnomalyScore",
    "robust_z",
    "volume_spike",
    "spread_shock",
    "volatility_explosion",
    "flow_extreme",
]

_MAD_SCALE: float = 0.6745
_MIN_SAMPLES: int = 10


@dataclass(frozen=True, slots=True, kw_only=True)
class AnomalyScore:
    """Un detector, un veredicto parcial (inmutable, explicable)."""

    name: str  # volume_spike | spread_shock | vol_explosion | flow_extreme | ...
    score: float  # z robusto (o ratio log); >= threshold ⇒ breach
    threshold: float
    breached: bool
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name no puede ser vacío")
        for field_name in ("score", "threshold"):
            if not math.isfinite(getattr(self, field_name)):
                raise ValueError(f"{field_name} no finito")


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def robust_z(series: tuple[float, ...] | list[float]) -> float:
    """z robusto del último punto vs la serie (0.0 sin dispersión)."""
    values = list(series)
    if len(values) < _MIN_SAMPLES:
        raise ValueError(
            f"serie insuficiente para z robusto: {len(values)} < {_MIN_SAMPLES}"
        )
    if any(not math.isfinite(v) for v in values):
        raise ValueError("serie no finita")
    med = _median(values)
    mad = _median([abs(v - med) for v in values])
    if mad <= 0:
        # Baseline sin dispersión mediana: fallback a z clásico; si ni
        # siquiera hay varianza, cualquier desvío es anomalía infinita.
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(var)
        if std <= 0:
            return float("inf") if values[-1] != med else 0.0
        return (values[-1] - mean) / std
    return (values[-1] - med) / (mad / _MAD_SCALE)


def volume_spike(volumes: tuple[float, ...] | list[float],
                 *, threshold: float = 4.0) -> AnomalyScore:
    """Volumen anormalmente alto en la última vela."""
    z = robust_z(tuple(volumes))
    return AnomalyScore(
        name="volume_spike", score=z, threshold=threshold,
        breached=z >= threshold,
        detail=f"z={z:.2f} vs umbral {threshold}",
    )


def spread_shock(spreads_bps: tuple[float, ...] | list[float],
                 *, threshold: float = 4.0) -> AnomalyScore:
    """Apertura anormal del spread (la liquidez se retira)."""
    z = robust_z(tuple(spreads_bps))
    return AnomalyScore(
        name="spread_shock", score=z, threshold=threshold,
        breached=z >= threshold,
        detail=f"z={z:.2f} vs umbral {threshold}",
    )


def volatility_explosion(returns: tuple[float, ...] | list[float],
                         *, baseline_n: int = 40, recent_n: int = 5,
                         threshold: float = 3.0) -> AnomalyScore:
    """Vol reciente / vol baseline (ratio; 1.0 = normal)."""
    values = tuple(returns)
    need = baseline_n + recent_n
    if len(values) < need:
        raise ValueError(
            f"retornos insuficientes: {len(values)} < {need}"
        )
    if any(not math.isfinite(r) for r in values):
        raise ValueError("retornos no finitos")

    def _std(xs: tuple[float, ...]) -> float:
        mean = sum(xs) / len(xs)
        return math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))

    base = _std(values[-(need):-recent_n]) if recent_n else _std(values)
    recent = _std(values[-recent_n:])
    ratio = recent / base if base > 0 else (float("inf") if recent > 0 else 1.0)
    score = math.log(ratio) / math.log(threshold) if ratio > 0 else 0.0
    return AnomalyScore(
        name="vol_explosion", score=score, threshold=1.0,
        breached=ratio >= threshold,
        detail=f"ratio={ratio:.2f}x vs umbral {threshold}x",
    )


def flow_extreme(imbalances: tuple[float, ...] | list[float],
                 *, threshold: float = 0.8) -> AnomalyScore:
    """Flujo unilateral extremo (|imbalance| reciente sobre umbral)."""
    values = tuple(imbalances)
    if len(values) < 3:
        raise ValueError(f"imbalances insuficientes: {len(values)} < 3")
    if any(not math.isfinite(v) for v in values):
        raise ValueError("imbalances no finitos")
    peak = max(abs(v) for v in values[-3:])
    return AnomalyScore(
        name="flow_extreme", score=peak, threshold=threshold,
        breached=peak >= threshold,
        detail=f"|imb|={peak:.2f} vs umbral {threshold}",
    )
