"""Random winner permutation test: destroys context→expert association."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from iot_machine_learning.domain.entities.market.replay.ablation import portfolio_net_returns
from .sig_utils import mean, percentile

__all__ = [
    "RandomWinnerResult",
    "random_winner_test",
]


@dataclass(frozen=True, slots=True)
class RandomWinnerResult:
    """Real vs ganador aleatorio por ventana (la selección importa?)."""

    real_mean: float
    null_mean: float
    null_std: float
    ci_low: float
    ci_high: float
    p_value: float
    n_permutations: int


def random_winner_test(
    windows: Sequence["PermWindow"],
    *,
    n_permutations: int = 1000,
    seed: int = 42,
) -> RandomWinnerResult:
    """La selección gana contra elegir el experto al azar?

    Nulo: por ventana, un experto aleatorio (uniforme entre los que
    tienen outcomes en el TEST) recibe peso 1.0. ``p_value`` es de una
    cola: fracción de nulos con agregado >= al real (¿seleccionar
    mejora o empeora?).
    """
    from .permutation import PermWindow
    rng = random.Random(seed)
    total_n = sum(w.n for w in windows)

    def aggregate(window_means: Sequence[float]) -> float:
        return (
            sum(m * w.n for m, w in zip(window_means, windows, strict=True)) / total_n
            if total_n
            else 0.0
        )

    real = aggregate(
        [
            mean(portfolio_net_returns(w.weights, w.per_timestamp, w.cost))
            if w.n
            else 0.0
            for w in windows
        ]
    )
    null_aggregates: list[float] = []
    for _ in range(n_permutations):
        window_means: list[float] = []
        for w in windows:
            if not w.n:
                window_means.append(0.0)
                continue
            candidates = sorted({str(e) for _, per in w.per_timestamp for e in per})
            if not candidates:
                window_means.append(0.0)
                continue
            winner = rng.choice(candidates)
            window_means.append(
                mean(
                    portfolio_net_returns(
                        {winner: 1.0}, w.per_timestamp, w.cost
                    )
                )
            )
        null_aggregates.append(aggregate(window_means))
    null_aggregates.sort()
    exceed = sum(1 for v in null_aggregates if v >= real)
    return RandomWinnerResult(
        real_mean=real,
        null_mean=mean(null_aggregates),
        null_std=(
            math.sqrt(
                sum((v - mean(null_aggregates)) ** 2 for v in null_aggregates)
                / max(len(null_aggregates) - 1, 1)
            )
        ),
        ci_low=percentile(null_aggregates, 0.025),
        ci_high=percentile(null_aggregates, 0.975),
        p_value=exceed / n_permutations,
        n_permutations=n_permutations,
    )