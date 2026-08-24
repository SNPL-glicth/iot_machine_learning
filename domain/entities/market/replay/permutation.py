"""Temporal permutation test: outcomes shuffled, predictions intact."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from iot_machine_learning.domain.entities.market.replay.ablation import portfolio_net_returns
from .sig_utils import mean, percentile

__all__ = [
    "PermWindow",
    "PermutationResult",
    "recover_predicted_direction",
    "permuted_net_returns",
    "permutation_test",
]


@dataclass(frozen=True, slots=True)
class PermWindow:
    """Una ventana con sus pesos y outcomes TEST (para permutar)."""

    weights: Mapping[str, float]
    per_timestamp: Sequence[tuple[float, Mapping[str, tuple[bool, float]]]]
    cost: float
    n: int


def recover_predicted_direction(correct: bool, move: float) -> int:
    """Dirección predicha (±1) recuperada de (direction_correct, move).

    Devuelve 0 cuando el movimiento es nulo (dirección indeterminada).
    """
    if move == 0.0:
        return 0
    sign = 1 if move > 0.0 else -1
    return sign if correct else -sign


def permuted_net_returns(
    weights: Mapping[str, float],
    per_timestamp: Sequence[tuple[float, Mapping[str, tuple[bool, float]]]],
    cost: float,
    rng: random.Random,
) -> list[float]:
    """PnL por timestamp con los OUTCOMES barajados temporalmente.

    La dirección predicha de cada experto se recupera desde su
    (correct, move) real y se re-evalúa contra el movimiento firmado
    de OTRO timestamp (la predicción queda intacta, el outcome se
    mueve en el tiempo). Bajo el nulo E[PnL] = 0 antes de costos.
    """
    active = {e: w for e, w in weights.items() if w > 0.0}
    entries: list[tuple[Mapping[str, tuple[bool, float]], float]] = []
    moves: list[float] = []
    for _, per_expert in per_timestamp:
        if not all(e in per_expert for e in active):
            continue
        move = next(iter(per_expert.values()))[1]
        if move == 0.0:
            continue
        entries.append((per_expert, move))
        moves.append(move)
    rng.shuffle(moves)
    returns: list[float] = []
    for (per_expert, _), shuffled_move in zip(entries, moves, strict=True):
        shuffled_sign = 1 if shuffled_move > 0.0 else -1
        pnl = 0.0
        for expert, weight in active.items():
            correct, move = per_expert[expert]
            predicted = recover_predicted_direction(correct, move)
            if predicted == 0:
                continue
            pnl += weight * (
                abs(shuffled_move) if predicted == shuffled_sign else -abs(shuffled_move)
            )
        returns.append(pnl - cost)
    return returns


@dataclass(frozen=True, slots=True)
class PermutationResult:
    """Resultado de la permutación temporal (por modo de selección)."""

    real_mean: float
    null_mean: float
    null_std: float
    ci_low: float
    ci_high: float
    p_value: float
    n_permutations: int


def permutation_test(
    windows: Sequence[PermWindow],
    *,
    n_permutations: int = 500,
    seed: int = 42,
) -> PermutationResult:
    """Permutación temporal sobre el agregado n-ponderado de las ventanas.

    ``p_value`` es de dos colas: fracción de permutaciones cuyo |agregado|
    alcanza o supera al real. Un p pequeño = el edge NO es explicable por
    la secuencia temporal de outcomes.
    """
    rng = random.Random(seed)

    def aggregate(
        window_means: Sequence[float],
    ) -> float:
        total_n = sum(w.n for w in windows)
        return (
            sum(m * w.n for m, w in zip(window_means, windows, strict=True)) / total_n
            if total_n
            else 0.0
        )

    real_means = [
        mean(portfolio_net_returns(w.weights, w.per_timestamp, w.cost))
        if w.n
        else 0.0
        for w in windows
    ]
    real = aggregate(real_means)
    null_aggregates: list[float] = []
    for _ in range(n_permutations):
        null_aggregates.append(
            aggregate(
                [
                    mean(permuted_net_returns(w.weights, w.per_timestamp, w.cost, rng))
                    if w.n
                    else 0.0
                    for w in windows
                ]
            )
        )
    null_aggregates.sort()
    exceed = sum(1 for v in null_aggregates if abs(v) >= abs(real))
    return PermutationResult(
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