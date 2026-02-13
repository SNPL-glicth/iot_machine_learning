"""Stability analysis and diagnostic assembly for the Taylor engine.

Pure functions — no I/O, no state, no logging.
"""

from __future__ import annotations

from typing import List

from .polynomial import compute_local_fit_error
from .types import TaylorCoefficients, TaylorDiagnostic

MIN_ACCEL_HISTORY: int = 4


def compute_accel_variance(values: List[float], dt: float) -> float:
    """Population variance of the second derivative across the window.

    Measures how much the acceleration changes — a proxy for
    how well a low-order polynomial approximates the signal.

    For a signal with additive noise σ:
        Var[f''] ≈ 6σ² / Δt⁴

    Args:
        values: Time series.
        dt: Time step.

    Returns:
        Population variance of f''. Returns 0.0 if < 4 points.
    """
    n = len(values)
    if n < MIN_ACCEL_HISTORY:
        return 0.0

    dt_sq = dt * dt
    accels: List[float] = []
    for i in range(2, n):
        accel = (values[i] - 2.0 * values[i - 1] + values[i - 2]) / dt_sq
        accels.append(accel)

    if len(accels) < 2:
        return 0.0

    mean_a = sum(accels) / len(accels)
    return sum((a - mean_a) ** 2 for a in accels) / len(accels)


def compute_stability_indicator(
    accel_variance: float, f_t: float,
) -> float:
    """Normalize acceleration variance into [0, 1].

    0.0 = perfectly stable (constant or linear signal).
    1.0 = highly unstable.

    Divides by |f(t)| (or 1.0 if near zero) and clamps to [0, 1].
    """
    normalizer = abs(f_t) if abs(f_t) > 1e-6 else 1.0
    return min(accel_variance / normalizer, 1.0)


def compute_diagnostic(
    coeffs: TaylorCoefficients,
    values: List[float],
    dt: float,
) -> TaylorDiagnostic:
    """Build a complete diagnostic for a Taylor prediction.

    Combines coefficient information with stability analysis
    and local fit error.

    Args:
        coeffs: Taylor coefficients from derivative estimation.
        values: Original time series (for variance / fit error).
        dt: Time step.

    Returns:
        ``TaylorDiagnostic`` with all diagnostic fields.
    """
    accel_var = compute_accel_variance(values, dt)
    stability = compute_stability_indicator(accel_var, coeffs.f_t)
    fit_error = compute_local_fit_error(coeffs, values, dt)

    return TaylorDiagnostic(
        estimated_order=coeffs.estimated_order,
        coefficients=coeffs.coefficients,
        local_slope=coeffs.local_slope,
        curvature=coeffs.curvature,
        stability_indicator=stability,
        accel_variance=accel_var,
        local_fit_error=fit_error,
        dt=dt,
        method=coeffs.method,
    )
