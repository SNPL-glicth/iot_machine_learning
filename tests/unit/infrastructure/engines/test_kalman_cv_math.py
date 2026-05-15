"""Tests for kalman_cv_math — pure 2D Constant-Velocity Kalman math.

Verifies initialization, prediction, update, and adaptive Q.
"""

from __future__ import annotations

import numpy as np
import pytest

from iot_machine_learning.infrastructure.ml.engines.kalman.kalman_cv_math import (
    KalmanCVState,
    adaptive_cv_update,
    initialize_cv_state,
    predict_cv,
    update_cv,
)


class TestKalmanCVMath:
    """Pure math tests — no I/O, no state persistence."""

    def test_cv_state_initializes_from_warmup(self) -> None:
        """Warmup of flat signal → x0 = mean, v0 = 0, R = var."""
        warmup = [10.0, 10.0, 10.0, 10.0, 10.0]
        state = initialize_cv_state(warmup, dt=1.0, Q_scale=1e-5)

        assert state.initialized is True
        assert state.x == pytest.approx(10.0, abs=1e-9)
        assert state.v == pytest.approx(0.0, abs=1e-9)
        # When variance is 0, R falls back to EPSILON.KALMAN_R
        assert state.R == pytest.approx(1e-6, abs=1e-9)
        assert state.P[0, 0] >= 0.0
        assert state.P[1, 1] >= 0.0
        assert state.dt == 1.0

    def test_cv_state_initializes_with_trend(self) -> None:
        """Warmup with linear trend → v0 ≈ slope."""
        warmup = [0.0, 1.0, 2.0, 3.0, 4.0]
        state = initialize_cv_state(warmup, dt=1.0, Q_scale=1e-5)

        assert state.initialized is True
        assert state.x == pytest.approx(2.0, abs=1e-9)
        assert state.v == pytest.approx(1.0, abs=1e-6)

    def test_cv_predict_updates_position(self) -> None:
        """predict_cv moves x by v*dt."""
        state = KalmanCVState(x=10.0, v=2.0, initialized=True, dt=1.0)
        state.P = np.eye(2) * 0.1
        state.Q = np.eye(2) * 1e-5

        pred = predict_cv(state, dt=1.0)

        assert pred.x == pytest.approx(12.0, abs=1e-9)
        assert pred.v == pytest.approx(2.0, abs=1e-9)
        # P should grow due to process noise
        assert pred.P[0, 0] > state.P[0, 0]

    def test_cv_predict_with_custom_dt(self) -> None:
        """predict_cv respects custom dt."""
        state = KalmanCVState(x=5.0, v=3.0, initialized=True, dt=2.0)
        state.P = np.eye(2) * 0.1
        state.Q = np.eye(2) * 1e-5

        pred = predict_cv(state, dt=2.0)

        assert pred.x == pytest.approx(11.0, abs=1e-9)  # 5 + 3*2
        assert pred.v == pytest.approx(3.0, abs=1e-9)

    def test_cv_update_reduces_uncertainty(self) -> None:
        """After update, position uncertainty P[0,0] < P_pred[0,0]."""
        state = KalmanCVState(x=10.0, v=0.0, initialized=True, dt=1.0)
        state.P = np.array([[1.0, 0.0], [0.0, 0.5]])
        state.Q = np.eye(2) * 1e-5
        state.R = 0.1

        pred = predict_cv(state, dt=1.0)
        p_pred_00 = pred.P[0, 0]

        updated = update_cv(pred, measurement=10.0)

        # Update with consistent measurement should reduce uncertainty
        assert updated.P[0, 0] < p_pred_00
        assert updated.x == pytest.approx(10.0, abs=0.5)

    def test_adaptive_q_increases_with_innovation(self) -> None:
        """Large innovations → Q grows."""
        warmup = [0.0, 1.0, 2.0, 3.0, 4.0]
        state = initialize_cv_state(warmup, dt=1.0, Q_scale=1e-5)

        q_before = state.Q[0, 0]

        # Feed measurements with HUGE innovations (signal jumps)
        for measurement in [100.0, 200.0, 300.0]:
            state = adaptive_cv_update(state, measurement)

        q_after = state.Q[0, 0]

        # Q should have grown significantly due to large innovations
        assert q_after > q_before * 10

    def test_adaptive_q_window_circular(self) -> None:
        """Innovation window respects _innovation_window_size."""
        warmup = [0.0, 1.0, 2.0, 3.0, 4.0]
        state = initialize_cv_state(warmup, dt=1.0, Q_scale=1e-5, innovation_window=5)

        for measurement in [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]:
            state = adaptive_cv_update(state, measurement)

        assert len(state._innovations) <= 5

    def test_minimal_warmup_single_point(self) -> None:
        """Single-point warmup should not crash."""
        state = initialize_cv_state([42.0], dt=1.0, Q_scale=1e-5)

        assert state.initialized is True
        assert state.x == pytest.approx(42.0, abs=1e-9)
        assert state.v == pytest.approx(0.0, abs=1e-9)

    def test_ols_slope_on_perfect_line(self) -> None:
        """OLS slope on y = 3t + 5 should return 3."""
        from iot_machine_learning.infrastructure.ml.engines.kalman.kalman_cv_math import (
            _compute_ols_slope,
        )

        values = [5.0, 8.0, 11.0, 14.0, 17.0]  # slope = 3
        slope = _compute_ols_slope(values, dt=1.0)

        assert slope == pytest.approx(3.0, abs=1e-9)
