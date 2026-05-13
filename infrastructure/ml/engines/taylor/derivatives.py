"""Derivative estimation methods for the Taylor engine.

Four methods with different accuracy/noise tradeoffs:

1. **Backward differences** — O(Δt), uses only past data, amplifies noise.
2. **Central differences** — O(Δt²), more accurate, symmetric stencil.
3. **Least-squares fit** — noise-resistant, local polynomial fit.
4. **Savitzky-Golay smoothed** (MATH-CRIT-1) — noise-resistant, adaptive window.

All functions are pure (no I/O, no state, no logging).
"""

from __future__ import annotations

import math
from typing import List, Optional

try:
    from scipy.signal import savgol_filter
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

from .least_squares import least_squares_fit as _ls_fit
from .types import DerivativeMethod, TaylorCoefficients


# MATH-CRIT-1: Configurable constants (no magic numbers)
_MIN_WINDOW_SIZE: int = 5
_DEFAULT_WINDOW_SIZE: int = 11
_POLYORDER: int = 3


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


def savitzky_golay_smoothed(
    values: List[float],
    dt: float,
    order: int,
    window_size: Optional[int] = None,
) -> TaylorCoefficients:
    """Savitzky-Golay smoothed derivatives (MATH-CRIT-1).
    
    Uses scipy.signal.savgol_filter to compute smoothed derivatives,
    reducing noise amplification from finite differences.
    
    Args:
        values: Time series values.
        dt: Time step between consecutive values.
        order: Taylor expansion order (1-3).
        window_size: Window size for filter (must be odd). If None, uses adaptive.
    
    Returns:
        TaylorCoefficients with smoothed derivatives.
    
    Fallback:
        If scipy not available or window too small, falls back to backward differences.
    
    Applies SRP: Smoothing is independent concern.
    Applies LSP: Can be used as drop-in replacement for backward_differences.
    """
    n = len(values)
    
    # Fallback if scipy not available
    if not _SCIPY_AVAILABLE:
        return backward_differences(values, dt, order)
    
    # Determine adaptive window size
    if window_size is None:
        window_size = min(_DEFAULT_WINDOW_SIZE, n)
    
    # Ensure window is odd and >= minimum
    if window_size % 2 == 0:
        window_size -= 1
    
    if window_size < _MIN_WINDOW_SIZE or n < _MIN_WINDOW_SIZE:
        # Fallback to backward differences
        return backward_differences(values, dt, order)
    
    try:
        # Smooth the signal (0th derivative)
        smoothed = savgol_filter(values, window_size, _POLYORDER, deriv=0)
        f_t = float(smoothed[-1])
        
        eff_order = min(order, 3)
        
        # First derivative
        f1 = 0.0
        if eff_order >= 1:
            first_deriv = savgol_filter(values, window_size, _POLYORDER, deriv=1, delta=dt)
            f1 = float(first_deriv[-1])
        
        # Second derivative
        f2 = 0.0
        if eff_order >= 2:
            second_deriv = savgol_filter(values, window_size, _POLYORDER, deriv=2, delta=dt)
            f2 = float(second_deriv[-1])
        
        # Third derivative (if requested)
        f3 = 0.0
        if eff_order >= 3:
            # Third derivative can be noisy even with smoothing
            # Use finite differences on smoothed second derivative
            if n >= 2:
                second_deriv = savgol_filter(values, window_size, _POLYORDER, deriv=2, delta=dt)
                f3 = (second_deriv[-1] - second_deriv[-2]) / dt
        
        # Validate results are finite
        if not all(math.isfinite(x) for x in [f_t, f1, f2, f3]):
            return backward_differences(values, dt, order)
        
        return TaylorCoefficients(
            f_t=f_t,
            f_prime=f1,
            f_double_prime=f2,
            f_triple_prime=f3,
            estimated_order=eff_order,
            method="savitzky_golay",
        )
    
    except Exception:
        # Any scipy error: fallback to backward differences
        return backward_differences(values, dt, order)


_METHODS = {
    DerivativeMethod.BACKWARD: backward_differences,
    DerivativeMethod.CENTRAL: central_differences,
    DerivativeMethod.LEAST_SQUARES: least_squares_fit,
    DerivativeMethod.SAVITZKY_GOLAY: savitzky_golay_smoothed,  # MATH-CRIT-1
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
