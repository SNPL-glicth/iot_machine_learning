"""Tests para infrastructure/ml/engines/taylor_math.py.

Verifica funciones matemáticas PURAS — sin I/O, sin estado.
"""

from __future__ import annotations

import pytest

from iot_machine_learning.infrastructure.ml.engines.taylor.math import (
    compute_accel_variance,
    compute_dt,
    compute_finite_differences,
    taylor_expand,
)


class TestComputeFiniteDifferences:
    """Tests para cálculo de derivadas por diferencias finitas."""

    def test_first_derivative_linear(self) -> None:
        """Serie lineal: f'(t) = 1.0 por paso."""
        derivs = compute_finite_differences([1.0, 2.0, 3.0, 4.0, 5.0], dt=1.0, order=1)
        assert derivs["f_t"] == 5.0
        assert derivs["f_prime"] == pytest.approx(1.0)

    def test_second_derivative_quadratic(self) -> None:
        """Serie cuadrática: f''(t) = 2.0."""
        # f(t) = t^2: [0, 1, 4, 9, 16]
        derivs = compute_finite_differences([0.0, 1.0, 4.0, 9.0, 16.0], dt=1.0, order=2)
        assert derivs["f_double_prime"] == pytest.approx(2.0)

    def test_third_derivative_cubic(self) -> None:
        """Serie cúbica: f'''(t) = 6.0."""
        # f(t) = t^3: [0, 1, 8, 27, 64]
        derivs = compute_finite_differences([0.0, 1.0, 8.0, 27.0, 64.0], dt=1.0, order=3)
        assert derivs["f_triple_prime"] == pytest.approx(6.0)

    def test_order_1_ignores_higher(self) -> None:
        derivs = compute_finite_differences([1.0, 2.0, 3.0], dt=1.0, order=1)
        assert derivs["f_double_prime"] == 0.0
        assert derivs["f_triple_prime"] == 0.0

    def test_insufficient_points_for_order(self) -> None:
        """Solo 2 puntos: no puede calcular f''."""
        derivs = compute_finite_differences([1.0, 2.0], dt=1.0, order=2)
        assert derivs["f_prime"] == pytest.approx(1.0)
        assert derivs["f_double_prime"] == 0.0

    def test_custom_dt(self) -> None:
        derivs = compute_finite_differences([0.0, 5.0], dt=0.5, order=1)
        assert derivs["f_prime"] == pytest.approx(10.0)


class TestTaylorExpand:
    """Tests para expansión de Taylor."""

    def test_order_0_returns_f_t(self) -> None:
        derivs = {"f_t": 10.0, "f_prime": 1.0, "f_double_prime": 0.0, "f_triple_prime": 0.0}
        assert taylor_expand(derivs, h=1.0, order=0) == 10.0

    def test_order_1_linear(self) -> None:
        derivs = {"f_t": 10.0, "f_prime": 2.0, "f_double_prime": 0.0, "f_triple_prime": 0.0}
        assert taylor_expand(derivs, h=1.0, order=1) == pytest.approx(12.0)

    def test_order_2_quadratic(self) -> None:
        derivs = {"f_t": 10.0, "f_prime": 2.0, "f_double_prime": 4.0, "f_triple_prime": 0.0}
        # 10 + 2*1 + 4*1/2 = 14
        assert taylor_expand(derivs, h=1.0, order=2) == pytest.approx(14.0)

    def test_order_3_cubic(self) -> None:
        derivs = {"f_t": 0.0, "f_prime": 0.0, "f_double_prime": 0.0, "f_triple_prime": 6.0}
        # 0 + 0 + 0 + 6*1/6 = 1
        assert taylor_expand(derivs, h=1.0, order=3) == pytest.approx(1.0)

    def test_horizon_scaling(self) -> None:
        derivs = {"f_t": 10.0, "f_prime": 2.0, "f_double_prime": 0.0, "f_triple_prime": 0.0}
        assert taylor_expand(derivs, h=3.0, order=1) == pytest.approx(16.0)


class TestComputeAccelVariance:
    """Tests para varianza de aceleración."""

    def test_constant_series_zero_variance(self) -> None:
        assert compute_accel_variance([5.0] * 10, dt=1.0) == pytest.approx(0.0)

    def test_linear_series_zero_variance(self) -> None:
        """Serie lineal: f'' = 0 en todos los puntos."""
        assert compute_accel_variance([1.0, 2.0, 3.0, 4.0, 5.0], dt=1.0) == pytest.approx(0.0)

    def test_insufficient_points(self) -> None:
        assert compute_accel_variance([1.0, 2.0, 3.0], dt=1.0) == 0.0

    def test_nonzero_variance(self) -> None:
        """Serie con aceleración variable."""
        values = [0.0, 1.0, 4.0, 5.0, 10.0]
        var = compute_accel_variance(values, dt=1.0)
        assert var > 0.0


class TestComputeDt:
    """Tests para cálculo de Δt."""

    def test_none_timestamps(self) -> None:
        assert compute_dt(None) == 1.0

    def test_single_timestamp(self) -> None:
        assert compute_dt([1.0]) == 1.0

    def test_uniform_timestamps(self) -> None:
        assert compute_dt([0.0, 1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_non_uniform_timestamps(self) -> None:
        """Mediana de [1, 1, 5] = 1."""
        dt = compute_dt([0.0, 1.0, 2.0, 7.0])
        assert dt == pytest.approx(1.0)

    def test_minimum_dt(self) -> None:
        """Δt nunca es menor que 1e-6."""
        dt = compute_dt([0.0, 0.0, 0.0])
        assert dt >= 1e-6
