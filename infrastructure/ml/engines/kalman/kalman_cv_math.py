"""Pure 2D Constant-Velocity Kalman math.

Implements a 2-state Kalman filter (position + velocity) using numpy
for 2×2 matrix operations.  Agnostic to domain — no I/O, no logging.

This is a NEW 2D implementation; it does NOT reuse kalman_math.py
(which is scalar 1D) to avoid mathematical hacks and regression risk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np

from core.parameters.numerical_constants import EPSILON

# Scale factor for adaptive Q (Mehra 1972 heuristic)
_ADAPTIVE_Q_SCALE: float = 0.1

# Bounds for adaptive Q scale factor
_MIN_Q_SCALE: float = 1e-8
_MAX_Q_SCALE: float = 1.0

# Default innovation window size for adaptive Q
_DEFAULT_INNOVATION_WINDOW: int = 20

# Constant matrices (avoid re-allocation per call)
_H = np.array([[1.0, 0.0]], dtype=float)
_I2 = np.eye(2, dtype=float)

# F cache by dt value (avoid re-allocation per predict call)
_F_CACHE: dict = {}


def _get_F(dt: float) -> np.ndarray:
    """Return cached state-transition matrix F for given dt."""
    if dt not in _F_CACHE:
        _F_CACHE[dt] = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
    return _F_CACHE[dt]


@dataclass
class KalmanCVState:
    """2D Constant-Velocity Kalman state.

    Attributes:
        x: Estimated position.
        v: Estimated velocity.
        P: 2×2 error covariance matrix [[Pxx, Pxv], [Pxv, Pvv]].
        Q: 2×2 process noise covariance.
        R: Measurement noise variance (scalar).
        initialized: True after warmup completed.
        _innovations: Circular buffer of recent innovations (for adaptive Q).
        _innovation_window_size: Window size for adaptive Q.
        dt: Time step between observations.
    """

    x: float = 0.0
    v: float = 0.0
    P: np.ndarray = field(default_factory=lambda: np.eye(2))
    Q: np.ndarray = field(default_factory=lambda: np.eye(2) * 1e-5)
    R: float = 1.0
    initialized: bool = False
    _innovations: List[float] = field(default_factory=list)
    _innovation_window_size: int = 0
    dt: float = 1.0


def _compute_process_noise_covariance(dt: float, q: float) -> np.ndarray:
    """Compute Q for continuous-white-noise acceleration model (CV).

    Q = q * [[dt³/3, dt²/2],
             [dt²/2, dt  ]]

    Args:
        dt: Time step.
        q: Process noise intensity (acceleration variance).

    Returns:
        2×2 process noise covariance matrix.
    """
    dt2 = dt * dt
    dt3 = dt2 * dt
    return q * np.array([[dt3 / 3.0, dt2 / 2.0], [dt2 / 2.0, dt]], dtype=float)


def _compute_ols_slope(values: List[float], dt: float) -> float:
    """OLS linear slope over equally-spaced data.

    Fits y = a + b·t where t_i = i·dt.

    Args:
        values: Observations.
        dt: Time step.

    Returns:
        Estimated slope b.
    """
    n = len(values)
    if n < 2:
        return 0.0
    t_mean = (n - 1) * dt / 2.0
    y_mean = sum(values) / n
    num = 0.0
    den = 0.0
    for i, y in enumerate(values):
        t_i = i * dt
        diff_t = t_i - t_mean
        num += diff_t * (y - y_mean)
        den += diff_t * diff_t
    if abs(den) < EPSILON.DIVISION:
        return 0.0
    return num / den


def initialize_cv_state(
    warmup_values: List[float],
    dt: float = 1.0,
    Q_scale: float = 1e-5,
    innovation_window: int = _DEFAULT_INNOVATION_WINDOW,
) -> KalmanCVState:
    """Create calibrated CV state from warmup window.

    Calibration:
    - x0 = mean(warmup_values)
    - v0 = OLS slope over warmup
    - P0[0,0] = var(warmup)  (position uncertainty)
    - P0[1,1] = var(slope)   (velocity uncertainty)
    - R = max(var(warmup), EPSILON.KALMAN_R)
    - Q from Q_scale and dt

    Args:
        warmup_values: Initial observations for calibration.
        dt: Time step between observations.
        Q_scale: Initial process noise intensity.
        innovation_window: Window size for adaptive Q.

    Returns:
        Initialized KalmanCVState.
    """
    n = len(warmup_values)
    x0 = sum(warmup_values) / n

    if n > 1:
        variance = sum((v - x0) ** 2 for v in warmup_values) / (n - 1)
    else:
        variance = 0.0

    v0 = _compute_ols_slope(warmup_values, dt)

    # Variance of slope estimator: σ² / Σ(t_i - t̄)²
    slope_denom = sum(
        (i * dt - (n - 1) * dt / 2.0) ** 2 for i in range(n)
    )
    if slope_denom > EPSILON.DIVISION and variance > 0:
        var_v = variance / slope_denom
    else:
        var_v = 0.0

    R_calibrated = max(variance, float(EPSILON.KALMAN_R))

    P0 = np.array(
        [[max(variance, float(EPSILON.KALMAN_P)), 0.0],
         [0.0, max(var_v, float(EPSILON.KALMAN_P))]],
        dtype=float,
    )

    Q0 = _compute_process_noise_covariance(dt, Q_scale)

    return KalmanCVState(
        x=x0,
        v=v0,
        P=P0,
        Q=Q0,
        R=R_calibrated,
        initialized=True,
        _innovations=[],
        _innovation_window_size=innovation_window,
        dt=dt,
    )


def predict_cv(state: KalmanCVState, dt: float) -> KalmanCVState:
    """Kalman prediction (a priori) for one time step.

    x_pred = F @ x
    P_pred = F @ P @ F^T + Q

    Args:
        state: Current state.
        dt: Time step (overrides state.dt if provided).

    Returns:
        New state after prediction (position/velocity updated, P grown).
    """
    F = _get_F(dt)

    x_vec = np.array([state.x, state.v], dtype=float)
    x_pred = F @ x_vec

    P_pred = F @ state.P @ F.T + state.Q

    # Ensure symmetry and positive-definite
    P_pred = (P_pred + P_pred.T) / 2.0

    return KalmanCVState(
        x=float(x_pred[0]),
        v=float(x_pred[1]),
        P=P_pred,
        Q=state.Q,
        R=state.R,
        initialized=state.initialized,
        _innovations=list(state._innovations),
        _innovation_window_size=state._innovation_window_size,
        dt=state.dt,
    )


def update_cv(state: KalmanCVState, measurement: float) -> KalmanCVState:
    """Kalman update (a posteriori) with a scalar measurement.

    y = z - H @ x_pred
    S = H @ P_pred @ H^T + R
    K = P_pred @ H^T / S
    x = x_pred + K * y
    P = (I - K @ H) @ P_pred

    Args:
        state: Predicted state (from predict_cv).
        measurement: Scalar observation z.

    Returns:
        Updated state after correction.
    """
    H = _H
    I2 = _I2

    x_vec = np.array([state.x, state.v], dtype=float)
    y = measurement - float(np.dot(H[0], x_vec))

    S = float((H @ state.P @ H.T)[0, 0]) + state.R
    if abs(S) < EPSILON.DIVISION:
        S = EPSILON.DIVISION

    K = (state.P @ H.T) / S
    K = K.reshape(2)

    x_new = x_vec + K * y
    P_new = (I2 - K.reshape(2, 1) @ H) @ state.P

    # Ensure symmetry and minimum floor
    P_new = (P_new + P_new.T) / 2.0
    P_new = np.maximum(P_new, EPSILON.KALMAN_P * np.eye(2))

    return KalmanCVState(
        x=float(x_new[0]),
        v=float(x_new[1]),
        P=P_new,
        Q=state.Q,
        R=state.R,
        initialized=state.initialized,
        _innovations=list(state._innovations),
        _innovation_window_size=state._innovation_window_size,
        dt=state.dt,
    )


def adaptive_cv_update(state: KalmanCVState, measurement: float) -> KalmanCVState:
    """Kalman update with adaptive Q based on innovation variance.

    Q is adjusted using the variance of recent innovations.
    When the series changes regime, innovations grow → Q increases →
    the filter tracks changes faster.

    Args:
        state: Current state (must have _innovation_window_size > 0).
        measurement: Scalar observation.

    Returns:
        Updated state with potentially adapted Q.
    """
    # Predict step
    state = predict_cv(state, state.dt)

    # Compute innovation before update
    innovation = measurement - state.x

    # Update innovation window
    state._innovations.append(innovation)
    win = state._innovation_window_size or _DEFAULT_INNOVATION_WINDOW
    if len(state._innovations) > win:
        state._innovations = state._innovations[-win:]

    # Adapt Q if enough innovations
    if len(state._innovations) >= 3:
        n = len(state._innovations)
        mean_innov = sum(state._innovations) / n
        var_innov = sum((v - mean_innov) ** 2 for v in state._innovations) / max(n - 1, 1)
        q = _ADAPTIVE_Q_SCALE * var_innov
        q = max(_MIN_Q_SCALE, min(_MAX_Q_SCALE, q))
        state.Q = _compute_process_noise_covariance(state.dt, q)

    # Standard update
    return update_cv(state, measurement)
