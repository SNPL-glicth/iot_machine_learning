"""Block bootstrap for window-level statistics."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from iot_machine_learning.domain.entities.market.replay.ablation import max_drawdown, sharpe_of
from .sig_utils import mean, percentile

__all__ = [
    "WindowRecord",
    "BootstrapCi",
    "weighted_acc",
    "weighted_net",
    "pooled_sharpe",
    "window_cumsum_maxdd",
    "block_bootstrap",
    "difference_ci",
]


@dataclass(frozen=True, slots=True)
class WindowRecord:
    """Datos de una ventana para remuestrear (nivel TEST real)."""

    n: int
    accuracy: float
    net: float
    returns: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class BootstrapCi:
    """Estimador puntual + IC 95% percentil de un bootstrap."""

    point: float
    ci_low: float
    ci_high: float
    n_boot: int

    @property
    def crosses_zero(self) -> bool:
        return self.ci_low <= 0.0 <= self.ci_high


def weighted_acc(records: Sequence[WindowRecord]) -> float:
    total_n = sum(r.n for r in records)
    return (
        sum(r.accuracy * r.n for r in records) / total_n if total_n else 0.0
    )


def weighted_net(records: Sequence[WindowRecord]) -> float:
    total_n = sum(r.n for r in records)
    return sum(r.net * r.n for r in records) / total_n if total_n else 0.0


def pooled_sharpe(records: Sequence[WindowRecord]) -> float:
    pooled: list[float] = []
    for r in records:
        pooled.extend(r.returns)
    return sharpe_of(pooled)


def window_cumsum_maxdd(records: Sequence[WindowRecord]) -> float:
    """maxDD del acumulado de las medias netas POR VENTANA (ver 9.3)."""
    return max_drawdown([r.net for r in records])


def block_bootstrap(
    records: Sequence[WindowRecord],
    *,
    statistic: Callable[[Sequence[WindowRecord]], float],
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> BootstrapCi:
    """IC percentil de un estadístico agregado por remuestreo de ventanas."""
    if not records:
        return BootstrapCi(0.0, 0.0, 0.0, n_boot)
    rng = random.Random(seed)
    point = statistic(records)
    distribution = sorted(
        statistic(rng.choices(records, k=len(records))) for _ in range(n_boot)
    )
    return BootstrapCi(
        point=point,
        ci_low=percentile(distribution, alpha / 2.0),
        ci_high=percentile(distribution, 1.0 - alpha / 2.0),
        n_boot=n_boot,
    )


def difference_ci(
    pairs: Sequence[tuple[float, float]],
    weights: Sequence[int],
    *,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> BootstrapCi:
    """IC 95% de la diferencia ZENIN − baseline por ventana (n-ponderada).

    El IC responde "¿ZENIN es mejor que la baseline?": si cruza cero,
    no hay superioridad demostrable con la evidencia actual.
    """
    if not pairs:
        return BootstrapCi(0.0, 0.0, 0.0, n_boot)
    rng = random.Random(seed)
    indices = list(range(len(pairs)))

    def statistic(drawn: Sequence[int]) -> float:
        total = 0.0
        total_n = 0
        for i in drawn:
            total += (pairs[i][0] - pairs[i][1]) * weights[i]
            total_n += weights[i]
        return total / total_n if total_n else 0.0

    point = statistic(indices)
    distribution = sorted(
        statistic(rng.choices(indices, k=len(indices))) for _ in range(n_boot)
    )
    return BootstrapCi(
        point=point,
        ci_low=percentile(distribution, alpha / 2.0),
        ci_high=percentile(distribution, 1.0 - alpha / 2.0),
        n_boot=n_boot,
    )