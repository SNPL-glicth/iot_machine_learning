"""Robust Δt estimation from timestamps.

Pure function — no I/O, no state, no logging.

Uses the **median** of consecutive positive differences for robustness
against occasional outlier gaps (e.g., sensor downtime).

Assumptions:
- If timestamps is None or has < 2 elements, Δt = 1.0 (index-based).
- Only positive diffs are considered (filters out-of-order points).
- Minimum returned value is EPSILON.CONFIDENCE to prevent division by zero
  in velocity/acceleration calculations (dt appears in denominators).
"""

from __future__ import annotations

from typing import List, Optional

from core.parameters.numerical_constants import EPSILON


def compute_dt(timestamps: Optional[List[float]]) -> float:
    """Estimate Δt from timestamps using the median of consecutive diffs.

    Args:
        timestamps: Unix timestamps (optional).

    Returns:
        Δt > 0. Minimum EPSILON.CONFIDENCE (1e-6) to prevent division by zero.
    """
    if timestamps is None or len(timestamps) < 2:
        return 1.0

    diffs = [
        timestamps[i] - timestamps[i - 1]
        for i in range(1, len(timestamps))
        if timestamps[i] > timestamps[i - 1]
    ]

    if not diffs:
        return 1.0

    diffs_sorted = sorted(diffs)
    mid = len(diffs_sorted) // 2
    if len(diffs_sorted) % 2 == 0:
        dt = (diffs_sorted[mid - 1] + diffs_sorted[mid]) / 2.0
    else:
        dt = diffs_sorted[mid]

    # Prevent division by zero in derivative calculations (velocity = Δvalue/Δt)
    return max(dt, EPSILON.CONFIDENCE)
