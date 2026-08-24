"""Sigmoid boost/reduce functions for bounded weight adjustments."""

from __future__ import annotations

import math


def sigmoid_boost(x: float, midpoint: float = 1.0, steepness: float = 2.0) -> float:
    """Bounded boost factor [1.0, 3.0) via sigmoid."""
    return 1.0 + 2.0 / (1.0 + math.exp(-steepness * (x - midpoint)))


def sigmoid_reduce(x: float, midpoint: float = 1.0, steepness: float = 2.0) -> float:
    """Bounded reduction factor (0.5, 1.0] via inverse sigmoid."""
    return 1.0 - 0.5 / (1.0 + math.exp(-steepness * (x - midpoint)))