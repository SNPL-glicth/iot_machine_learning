"""Funciones de scoring puras. Sin estado, sin I/O, sin sklearn."""
from __future__ import annotations


def compute_z_score(value: float, mean: float, std: float) -> float:
    if std < 1e-12:
        return 0.0
    return abs(value - mean) / std


def compute_z_vote(
    z_score: float,
    lower: float = 2.0,
    upper: float = 3.0,
) -> float:
    if z_score > upper:
        return 1.0
    if z_score > lower:
        span = upper - lower
        return (z_score - lower) / span if span > 0 else 1.0
    return 0.0


def compute_iqr_bounds(
    q1: float, q3: float, iqr: float
) -> tuple[float, float]:
    return (q1 - 2.5 * iqr, q3 + 2.5 * iqr)


def weighted_vote(
    votes: dict,
    weights: dict,
    default_weight: float = 0.1,
) -> float:
    tot = sum(weights.get(m, default_weight) for m in votes)
    return sum(votes[m] * weights.get(m, default_weight) for m in votes) / max(tot, 1e-12)


def compute_iqr_vote(v: float, q1: float, q3: float, iqr: float) -> float:
    return 1.0 if (v < q1 - 2.5 * iqr or v > q3 + 2.5 * iqr) else 0.0


def compute_consensus_confidence(votes: dict) -> float:
    return 1.0
