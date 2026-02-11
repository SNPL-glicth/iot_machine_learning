"""Tests para infrastructure/ml/filters/kalman_math.py.

Verifica funciones matemáticas PURAS — sin I/O, sin threading.
"""

from __future__ import annotations

import pytest

from iot_machine_learning.infrastructure.ml.filters.kalman_math import (
    KalmanState,
    WarmupBuffer,
    initialize_state,
    kalman_update,
    MIN_R,
    MIN_P,
)


class TestKalmanState:
    """Tests para dataclass KalmanState."""

    def test_default_values(self) -> None:
        state = KalmanState()
        assert state.x_hat == 0.0
        assert state.P == 1.0
        assert state.initialized is False

    def test_custom_values(self) -> None:
        state = KalmanState(x_hat=20.0, P=0.5, Q=1e-4, R=0.1, initialized=True)
        assert state.x_hat == 20.0
        assert state.initialized is True


class TestWarmupBuffer:
    """Tests para WarmupBuffer."""

    def test_not_ready_initially(self) -> None:
        buf = WarmupBuffer(target_size=5)
        assert buf.is_ready is False

    def test_ready_when_full(self) -> None:
        buf = WarmupBuffer(values=[1.0, 2.0, 3.0, 4.0, 5.0], target_size=5)
        assert buf.is_ready is True

    def test_ready_when_over(self) -> None:
        buf = WarmupBuffer(values=[1.0] * 10, target_size=5)
        assert buf.is_ready is True


class TestInitializeState:
    """Tests para calibración de estado desde warmup."""

    def test_mean_calibration(self) -> None:
        state = initialize_state([20.0, 20.0, 20.0, 20.0, 20.0], Q=1e-5)
        assert state.x_hat == pytest.approx(20.0)
        assert state.initialized is True

    def test_r_calibration_from_variance(self) -> None:
        """R se calibra como varianza de las lecturas de warmup."""
        values = [19.0, 20.0, 21.0, 20.0, 19.0]
        state = initialize_state(values, Q=1e-5)
        assert state.R > MIN_R

    def test_r_minimum_clamp(self) -> None:
        """Valores idénticos → R clamped a MIN_R."""
        state = initialize_state([5.0, 5.0, 5.0], Q=1e-5)
        assert state.R == pytest.approx(MIN_R)

    def test_p_minimum_clamp(self) -> None:
        """P nunca baja de MIN_P."""
        state = initialize_state([5.0, 5.0, 5.0], Q=1e-5)
        assert state.P >= MIN_P

    def test_q_preserved(self) -> None:
        state = initialize_state([1.0, 2.0, 3.0], Q=0.01)
        assert state.Q == 0.01


class TestKalmanUpdate:
    """Tests para paso de Kalman update."""

    def test_converges_to_measurement(self) -> None:
        """Con muchas iteraciones, x_hat converge al valor medido."""
        state = KalmanState(x_hat=0.0, P=1.0, Q=1e-5, R=0.1, initialized=True)
        for _ in range(200):
            kalman_update(state, 20.0)
        assert abs(state.x_hat - 20.0) < 0.01

    def test_filters_noise(self) -> None:
        """Kalman suaviza ruido alrededor de un valor constante."""
        import random
        random.seed(42)
        state = KalmanState(x_hat=20.0, P=1.0, Q=1e-5, R=1.0, initialized=True)
        for _ in range(50):
            noisy = 20.0 + random.gauss(0, 1.0)
            kalman_update(state, noisy)
        assert abs(state.x_hat - 20.0) < 0.5

    def test_p_decreases(self) -> None:
        """P (incertidumbre) debe decrecer con cada update."""
        state = KalmanState(x_hat=20.0, P=1.0, Q=1e-5, R=0.1, initialized=True)
        p_initial = state.P
        kalman_update(state, 20.0)
        assert state.P < p_initial

    def test_returns_filtered_value(self) -> None:
        state = KalmanState(x_hat=20.0, P=1.0, Q=1e-5, R=0.1, initialized=True)
        result = kalman_update(state, 25.0)
        assert 20.0 < result < 25.0

    def test_state_modified_in_place(self) -> None:
        state = KalmanState(x_hat=10.0, P=1.0, Q=1e-5, R=0.1, initialized=True)
        kalman_update(state, 20.0)
        assert state.x_hat != 10.0
