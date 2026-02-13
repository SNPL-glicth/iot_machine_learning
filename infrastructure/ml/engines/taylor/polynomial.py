"""Taylor polynomial projection and local fit error.

Pure functions — no I/O, no state, no logging.

Mathematical basis:
    P(h) = f(t) + f'(t)·h + f''(t)·h²/2! + f'''(t)·h³/3!

The ``project`` function evaluates this polynomial at a given
horizon *h* (in time units).

``compute_local_fit_error`` measures how well the polynomial
reconstructs the recent data — a proxy for prediction reliability.
"""

from __future__ import annotations

import math
from typing import List

from .types import TaylorCoefficients


def project(coeffs: TaylorCoefficients, h: float, order: int) -> float:
    """Evaluate the Taylor polynomial at t + h.

    P(h) = f(t) + f'(t)·h + f''(t)·h²/2! + f'''(t)·h³/3!

    Only terms up to ``order`` are included.

    Args:
        coeffs: Taylor coefficients from derivative estimation.
        h: Projection horizon (in time units, typically horizon × Δt).
        order: Maximum order to include (1–3).

    Returns:
        Predicted value f(t + h).
    """
    result = coeffs.f_t

    if order >= 1:
        result += coeffs.f_prime * h
    if order >= 2:
        result += coeffs.f_double_prime * (h * h) / 2.0
    if order >= 3:
        result += coeffs.f_triple_prime * (h * h * h) / 6.0

    return result


def compute_local_fit_error(
    coeffs: TaylorCoefficients,
    values: List[float],
    dt: float,
    n_points: int = 5,
) -> float:
    """RMS error of the Taylor polynomial over the last ``n_points``.

    Evaluates how well the polynomial centered at the last point
    reconstructs the recent history.  A high fit error suggests the
    signal is not well-approximated by a low-order polynomial.

    Args:
        coeffs: Taylor coefficients.
        values: Full time series.
        dt: Time step.
        n_points: Number of recent points to check (default 5).

    Returns:
        RMS residual.  Returns 0.0 if insufficient data.
    """
    n = len(values)
    check = min(n_points, n)
    if check < 2:
        return 0.0

    order = coeffs.estimated_order
    sum_sq = 0.0

    for i in range(check):
        idx = n - check + i
        h = (idx - (n - 1)) * dt  # negative for past points, 0 for last
        predicted = project(coeffs, h, order)
        residual = values[idx] - predicted
        sum_sq += residual * residual

    return math.sqrt(sum_sq / check)
