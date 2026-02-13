"""Derivative estimation methods for the Taylor engine.

Three methods with different accuracy/noise tradeoffs:

1. **Backward differences** — O(Δt), uses only past data, amplifies noise.
2. **Central differences** — O(Δt²), more accurate, symmetric stencil.
3. **Least-squares fit** — noise-resistant, local polynomial fit.

All functions are pure (no I/O, no state, no logging).
"""

from __future__ import annotations

from typing import List

from .least_squares import least_squares_fit as _ls_fit
from .types import DerivativeMethod, TaylorCoefficients


def backward_differences(
    values: List[float], dt: float, order: int,
) -> TaylorCoefficients:
    """Backward finite differences at the last point.

    Formulas (O(Δt) accuracy):
        f'(t)   = [f(t) − f(t−Δt)] / Δt
        f''(t)  = [f(t) − 2f(t−Δt) + f(t−2Δt)] / Δt²
        f'''(t) = [f(t) − 3f(t−Δt) + 3f(t−2Δt) − f(t−3Δt)] / Δt³
    """
    n = len(values)
    f_t = values[-1]
    eff = min(order, 3)

    f1 = 0.0
    if eff >= 1 and n >= 2:
        f1 = (values[-1] - values[-2]) / dt
    else:
        eff = 0

    f2 = 0.0
    if eff >= 2 and n >= 3:
        f2 = (values[-1] - 2.0 * values[-2] + values[-3]) / (dt * dt)
    elif eff >= 2:
        eff = 1

    f3 = 0.0
    if eff >= 3 and n >= 4:
        f3 = (values[-1] - 3.0 * values[-2]
              + 3.0 * values[-3] - values[-4]) / (dt ** 3)
    elif eff >= 3:
        eff = 2

    return TaylorCoefficients(
        f_t=f_t, f_prime=f1, f_double_prime=f2,
        f_triple_prime=f3, estimated_order=eff, method="backward",
    )


def central_differences(
    values: List[float], dt: float, order: int,
) -> TaylorCoefficients:
    """Central finite differences — O(Δt²) accuracy.

    Formulas:
        f'(t)   = [f(t+Δt) − f(t−Δt)] / (2Δt)
        f''(t)  = [f(t+Δt) − 2f(t) + f(t−Δt)] / Δt²
        f'''(t) = [f(t+2) − 2f(t+1) + 2f(t−1) − f(t−2)] / (2Δt³)
    """
    n = len(values)
    f_t = values[-1]
    eff = min(order, 3)

    f1 = 0.0
    if eff >= 1 and n >= 3:
        f1 = (values[-1] - values[-3]) / (2.0 * dt)
    elif eff >= 1 and n >= 2:
        f1 = (values[-1] - values[-2]) / dt
        eff = min(eff, 1)
    else:
        eff = 0

    f2 = 0.0
    if eff >= 2 and n >= 3:
        f2 = (values[-1] - 2.0 * values[-2] + values[-3]) / (dt * dt)
    elif eff >= 2:
        eff = 1

    f3 = 0.0
    if eff >= 3 and n >= 5:
        f3 = (values[-1] - 2.0 * values[-2]
              + 2.0 * values[-4] - values[-5]) / (2.0 * dt ** 3)
    elif eff >= 3:
        eff = 2

    return TaylorCoefficients(
        f_t=f_t, f_prime=f1, f_double_prime=f2,
        f_triple_prime=f3, estimated_order=eff, method="central",
    )


def least_squares_fit(
    values: List[float], dt: float, order: int,
) -> TaylorCoefficients:
    """Local polynomial least-squares fit — delegates to least_squares.py.

    Falls back to backward differences if insufficient data or singular.
    """
    result = _ls_fit(values, dt, order)
    if result is None:
        return backward_differences(values, dt, order)
    return result


_METHODS = {
    DerivativeMethod.BACKWARD: backward_differences,
    DerivativeMethod.CENTRAL: central_differences,
    DerivativeMethod.LEAST_SQUARES: least_squares_fit,
}


def estimate_derivatives(
    values: List[float],
    dt: float,
    order: int,
    method: DerivativeMethod = DerivativeMethod.BACKWARD,
) -> TaylorCoefficients:
    """Dispatch to the requested derivative estimation method."""
    fn = _METHODS.get(method, backward_differences)
    return fn(values, dt, order)
