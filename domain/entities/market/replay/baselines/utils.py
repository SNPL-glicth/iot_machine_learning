"""Baselines — Utility functions."""

from __future__ import annotations

import math


def logistic(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def clamp_prob(value: float) -> float:
    return min(0.95, max(0.05, value))


def band(expected: float, vol: float, horizon_ratio: float) -> tuple[float, float]:
    half = 0.6745 * vol * math.sqrt(horizon_ratio)
    return expected - half, expected + half