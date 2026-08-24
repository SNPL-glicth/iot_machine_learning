"""FASE 10.1 + 10.5 — Calibration Metrics (Brier, ECE, LogLoss, Wilson, Economic)."""

from __future__ import annotations

import math


def compute_brier(probs: list[float], outcomes: list[bool]) -> float:
    """Calcula Brier score."""
    if not probs:
        return 0.0
    errors = []
    for prob, outcome in zip(probs, outcomes):
        y = 1.0 if outcome else 0.0
        errors.append((prob - y) ** 2)
    return sum(errors) / len(errors)


def compute_ece(
    probs: list[float],
    outcomes: list[bool],
    n_bins: int = 10,
) -> float:
    """Calcula Expected Calibration Error."""
    if not probs:
        return 0.0
    
    # Crear bins
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for prob, outcome in zip(probs, outcomes):
        bin_idx = min(int(prob * n_bins), n_bins - 1)
        bins[bin_idx].append((prob, outcome))
    
    # Calcular ECE
    ece = 0.0
    total = len(probs)
    
    for bin_items in bins:
        if not bin_items:
            continue
        n = len(bin_items)
        avg_prob = sum(prob for prob, _ in bin_items) / n
        accuracy = sum(1 for _, outcome in bin_items if outcome) / n
        ece += n / total * abs(avg_prob - accuracy)
    
    return ece


def compute_log_loss(probabilities: list[float], outcomes: list[bool]) -> float:
    """Calcula Log Loss."""
    if not probabilities:
        return 0.0
    
    total = 0.0
    for prob, outcome in zip(probabilities, outcomes):
        y = 1.0 if outcome else 0.0
        prob = max(1e-15, min(1 - 1e-15, prob))
        total -= y * math.log(prob) + (1 - y) * math.log(1 - prob)
    
    return total / len(probabilities)


def compute_wilson_lb(hits: int, n: int, z: float = 1.96) -> float:
    """Wilson score interval lower bound for binomial proportion.
    
    Args:
        hits: Number of successes
        n: Total trials
        z: Z-score (1.96 for 95% CI)
    
    Returns:
        Lower bound of Wilson interval
    """
    if n == 0:
        return 0.0
    p = hits / n
    denominator = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    adjustment = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return max(0.0, (center - adjustment) / denominator)


def compute_economic_edge(
    probabilities: list[float],
    outcomes: list[bool],
    returns: list[float] | None = None,
) -> float:
    """Calcula edge económico: expected value of betting on calibrated probabilities.
    
    Simplified: assumes binary betting with fixed payoff.
    If returns provided, uses actual PnL per prediction.
    """
    if not probabilities:
        return 0.0
    
    if returns is not None and len(returns) == len(probabilities):
        # Use actual returns
        edge = sum(p * r for p, r in zip(probabilities, returns)) / len(probabilities)
        return edge
    
    # Binary betting: bet 1 unit, win 1 if correct, lose 1 if wrong
    # Edge = P(win) * 1 + P(lose) * (-1) = 2 * accuracy - 1
    hits = sum(1 for p, o in zip(probabilities, outcomes) if (p >= 0.5) == o)
    accuracy = hits / len(probabilities)
    return 2 * accuracy - 1