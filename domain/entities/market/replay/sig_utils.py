"""Utility functions for statistical significance tests."""

from __future__ import annotations

import math
from collections.abc import Sequence


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Percentil q ∈ [0, 1] con interpolación lineal sobre valores ordenados."""
    if not sorted_values:
        return 0.0
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac