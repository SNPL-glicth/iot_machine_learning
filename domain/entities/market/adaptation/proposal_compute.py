"""Proposal computation helpers (softmax, bounded update)."""

from __future__ import annotations

import math
from typing import Mapping

from .expert_scores import ExpertScore


def default_weights(expert_names: Iterable[str], floor: float = 0.05) -> dict[str, float]:
    """Pesos uniformes de arranque (model v1) para los expertos dados."""
    names = tuple(expert_names)
    if not names:
        raise ValueError("expert_names no puede estar vacío")
    if not 0.0 <= floor <= 1.0 / len(names):
        raise ValueError(f"floor inválido para {len(names)} expertos: {floor}")
    share = 1.0 / len(names)
    return {name: share for name in names}


def softmax_target(
    eligible: list[ExpertScore],
    temperature: float,
) -> dict[str, float]:
    exponents = {s.expert: s.reward_adjusted / temperature for s in eligible}
    max_exp = max(exponents.values())
    exps = {name: math.exp(v - max_exp) for name, v in exponents.items()}
    total = sum(exps.values())
    return {name: v / total for name, v in exps.items()}


def bounded_update(
    current: Mapping[str, float],
    target: Mapping[str, float],
    *,
    max_change: float,
    min_weight: float,
    default_weight: float,
) -> dict[str, float]:
    """Acota |Δ| <= max_change, aplica piso y suma exacta 1.

    Los tres constraints pueden chocar (softmax ambicioso + piso +
    cota): se resuelve por redistribución iterativa determinista —
    el excedente se reparte entre los expertos con margen para bajar,
    proporcional a su margen (Δ + max_change). Si no hay solución
    factible, se conservan los pesos actuales (no cambiar es seguro).
    """
    if not target:
        return dict(current)
    merged = dict(current)
    for expert, weight in target.items():
        merged[expert] = weight
    base = {e: current.get(e, default_weight) for e in merged}
    deltas = {e: merged[e] - base[e] for e in merged}

    for _ in range(200):
        for e in deltas:
            deltas[e] = min(max(deltas[e], -max_change), max_change)
        weights = {e: base[e] + deltas[e] for e in merged}
        floored: dict[str, bool] = {}
        for e, w in weights.items():
            if w < min_weight:
                weights[e] = min_weight
                floored[e] = True
        excess = sum(weights.values()) - 1.0
        if abs(excess) <= 1e-12:
            return weights
        if excess > 0:
            candidates = [
                e for e in merged if not floored.get(e) and deltas[e] > -max_change + 1e-12
            ]
        else:
            candidates = [
                e for e in merged if not floored.get(e) and deltas[e] < max_change - 1e-12
            ]
        if not candidates:
            return dict(base)
        margin = sum(
            deltas[e] + max_change if excess > 0 else max_change - deltas[e]
            for e in candidates
        )
        if margin <= 0:
            return dict(base)
        for e in candidates:
            room = deltas[e] + max_change if excess > 0 else max_change - deltas[e]
            deltas[e] -= excess * (room / margin)
    return weights


def reason(score: ExpertScore, delta: float) -> str:
    direction = "increase" if delta > 0 else "decrease"
    return (
        f"{direction} under {score.regime or 'ALL'}/{score.horizon_seconds}s: "
        f"reward_adjusted {score.reward_adjusted:+.4f} "
        f"(reward {score.mean_reward:+.4f}, cal {score.calibration_error:.2f}, "
        f"acc {score.accuracy:.1%}, n={score.n})"
    )