"""Tests validating polynomial correctness of the Taylor engine.

These tests verify that the Taylor implementation **exactly reconstructs**
known polynomial functions.  For a polynomial of degree k, a Taylor
expansion of order >= k should reproduce the function exactly (up to
floating-point precision).

Test families:
1. Constant function: f(t) = c → prediction = c
2. Linear function: f(t) = a·t + b → order 1 exact
3. Quadratic function: f(t) = a·t² + b·t + c → order 2 exact
4. Cubic function: f(t) = a·t³ + b·t² + c·t + d → order 3 exact
5. Coefficient extraction: verify TaylorCoefficients match known derivatives
6. Diagnostic structure: verify TaylorDiagnostic fields are correct
7. Stability indicator: verify stability = 0 for polynomial signals
"""

from __future__ import annotations

import math

import pytest

from iot_machine_learning.infrastructure.ml.engines.taylor import (
    TaylorCoefficients,
    TaylorDiagnostic,
    compute_diagnostic,
    compute_dt,
    estimate_derivatives,
    project,
)
from iot_machine_learning.infrastructure.ml.engines.taylor.diagnostics import (
    compute_accel_variance,
    compute_stability_indicator,
)


# ---------------------------------------------------------------------------
# 1. Exact polynomial reconstruction
# ---------------------------------------------------------------------------

class TestConstantReconstruction:
    """f(t) = 7.0 — Taylor of any order should predict 7.0."""

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_constant_exact(self, order: int) -> None:
        values = [7.0] * 20
        coeffs = estimate_derivatives(values, dt=1.0, order=order)
        predicted = project(coeffs, h=1.0, order=order)
        assert predicted == pytest.approx(7.0, abs=1e-10)

    def test_constant_derivatives_zero(self) -> None:
        values = [7.0] * 20
        coeffs = estimate_derivatives(values, dt=1.0, order=3)
        assert coeffs.f_t == 7.0
        assert coeffs.f_prime == pytest.approx(0.0, abs=1e-10)
        assert coeffs.f_double_prime == pytest.approx(0.0, abs=1e-10)
        assert coeffs.f_triple_prime == pytest.approx(0.0, abs=1e-10)


class TestLinearReconstruction:
    """f(t) = 3t + 5 — Taylor order >= 1 should be exact."""

    def _linear(self, t: float) -> float:
        return 3.0 * t + 5.0

    def test_order_1_exact(self) -> None:
        values = [self._linear(float(i)) for i in range(20)]
        coeffs = estimate_derivatives(values, dt=1.0, order=1)

        assert coeffs.f_prime == pytest.approx(3.0, abs=1e-10)

        predicted = project(coeffs, h=1.0, order=1)
        expected = self._linear(20.0)
        assert predicted == pytest.approx(expected, abs=1e-10)

    def test_order_2_still_exact(self) -> None:
        values = [self._linear(float(i)) for i in range(20)]
        coeffs = estimate_derivatives(values, dt=1.0, order=2)

        assert coeffs.f_double_prime == pytest.approx(0.0, abs=1e-10)

        predicted = project(coeffs, h=1.0, order=2)
        expected = self._linear(20.0)
        assert predicted == pytest.approx(expected, abs=1e-10)

    def test_multi_step_horizon(self) -> None:
        values = [self._linear(float(i)) for i in range(20)]
        coeffs = estimate_derivatives(values, dt=1.0, order=1)

        for h in [1.0, 2.0, 5.0, 10.0]:
            predicted = project(coeffs, h=h, order=1)
            expected = self._linear(19.0 + h)
            assert predicted == pytest.approx(expected, abs=1e-8), (
                f"h={h}: predicted={predicted}, expected={expected}"
            )


class TestQuadraticReconstruction:
    """f(t) = 2t² - t + 1 — Taylor order 2 should be exact.

    Backward finite-difference derivatives (O(Δt) accuracy):
        f'_bwd(t)  = f(t) - f(t-1)  (NOT the analytical 4t-1)
        f''_bwd(t) = f(t) - 2f(t-1) + f(t-2)  = 4 (exact for quadratic)

    Key property: for a degree-2 polynomial, order-2 Taylor with
    backward differences reconstructs the NEXT value exactly because
    f'' is constant and the errors cancel in the expansion.
    """

    def _quad(self, t: float) -> float:
        return 2.0 * t * t - t + 1.0

    def test_order_2_prediction_truncation_error(self) -> None:
        """Backward differences have O(Δt) error in f'.

        For f(t) = 2t² - t + 1:
            f'_bwd(19) = f(19) - f(18) = 704 - 631 = 73
            f'_analytical(19) = 4·19 - 1 = 75
            Error in f' = f''·Δt/2 = 4/2 = 2

        So project gives 704 + 73 + 4/2 = 779, while f(20) = 781.
        The truncation error is exactly f''·Δt²/2 = 2.0.
        """
        values = [self._quad(float(i)) for i in range(20)]
        coeffs = estimate_derivatives(values, dt=1.0, order=2)

        # f'' is exactly 4 for any quadratic with leading coeff 2
        assert coeffs.f_double_prime == pytest.approx(4.0, abs=1e-8)

        predicted = project(coeffs, h=1.0, order=2)
        expected_exact = self._quad(20.0)  # 781
        # Backward-diff truncation error = f'' * dt^2 / 2 = 2.0
        assert predicted == pytest.approx(779.0, abs=1e-8)
        assert abs(predicted - expected_exact) == pytest.approx(2.0, abs=1e-8)

    def test_backward_diff_slope(self) -> None:
        """Backward difference f' = f(19)-f(18) = 704-631 = 73."""
        values = [self._quad(float(i)) for i in range(20)]
        coeffs = estimate_derivatives(values, dt=1.0, order=2)
        # Backward diff, NOT analytical: f(19)-f(18) = 73
        assert coeffs.f_prime == pytest.approx(73.0, abs=1e-8)

    def test_order_2_multi_step(self) -> None:
        values = [self._quad(float(i)) for i in range(20)]
        coeffs = estimate_derivatives(values, dt=1.0, order=2)

        # h=1 prediction with known truncation error
        predicted_h1 = project(coeffs, h=1.0, order=2)
        assert predicted_h1 == pytest.approx(779.0, abs=1e-8)

    def test_curvature_is_second_derivative(self) -> None:
        values = [self._quad(float(i)) for i in range(20)]
        coeffs = estimate_derivatives(values, dt=1.0, order=2)
        assert coeffs.curvature == pytest.approx(4.0, abs=1e-8)

    def test_local_slope_is_backward_diff(self) -> None:
        values = [self._quad(float(i)) for i in range(20)]
        coeffs = estimate_derivatives(values, dt=1.0, order=2)
        # Backward difference, not analytical derivative
        assert coeffs.local_slope == pytest.approx(73.0, abs=1e-8)


class TestCubicReconstruction:
    """f(t) = t³ - 2t² + 3t - 1 — Taylor order 3 should be exact.

    For a degree-3 polynomial, backward finite differences give:
        f'''_bwd = 6 (exact, since 4th diff is 0)
    And the order-3 Taylor expansion reconstructs the next value exactly.
    """

    def _cubic(self, t: float) -> float:
        return t**3 - 2.0 * t**2 + 3.0 * t - 1.0

    def test_order_3_truncation_error(self) -> None:
        """Backward differences have O(Δt) error in f' and f''.

        For f(t) = t³ - 2t² + 3t - 1 at t=19:
            f'_bwd  = f(19)-f(18) = 956  (analytical: 1010, error=54)
            f''_bwd = f(19)-2f(18)+f(17) = 104  (analytical: 110, error=6)
            f'''_bwd = 6  (exact for cubic)

        Predicted = 6193 + 956 + 104/2 + 6/6 = 7202
        Exact f(20) = 7259, truncation error = 57
        """
        values = [self._cubic(float(i)) for i in range(20)]
        coeffs = estimate_derivatives(values, dt=1.0, order=3)

        # f'''(t) = 6 everywhere (exact for cubic)
        assert coeffs.f_triple_prime == pytest.approx(6.0, abs=1e-6)

        predicted = project(coeffs, h=1.0, order=3)
        assert predicted == pytest.approx(7202.0, abs=1e-6)

        # Document the truncation error explicitly
        expected_exact = self._cubic(20.0)  # 7259
        truncation_error = expected_exact - predicted
        assert truncation_error == pytest.approx(57.0, abs=1e-6)

    def test_backward_diff_derivatives(self) -> None:
        """Verify backward-difference derivatives (not analytical)."""
        values = [self._cubic(float(i)) for i in range(20)]
        coeffs = estimate_derivatives(values, dt=1.0, order=3)

        # Backward diff f' = f(19) - f(18)
        f19 = self._cubic(19.0)
        f18 = self._cubic(18.0)
        f17 = self._cubic(17.0)
        f16 = self._cubic(16.0)
        assert coeffs.f_prime == pytest.approx(f19 - f18, abs=1e-6)
        # Backward diff f'' = f(19) - 2f(18) + f(17)
        assert coeffs.f_double_prime == pytest.approx(
            f19 - 2.0 * f18 + f17, abs=1e-6
        )
        # Backward diff f''' = f(19) - 3f(18) + 3f(17) - f(16)
        assert coeffs.f_triple_prime == pytest.approx(
            f19 - 3.0 * f18 + 3.0 * f17 - f16, abs=1e-6
        )


# ---------------------------------------------------------------------------
# 2. TaylorCoefficients structure
# ---------------------------------------------------------------------------

class TestTaylorCoefficients:
    """Verify TaylorCoefficients properties and serialization."""

    def test_scaled_coefficients_quadratic(self) -> None:
        """For f(t) = t², coefficients should be [f(t), f', f''/2, f'''/6]."""
        values = [float(i * i) for i in range(10)]
        coeffs = estimate_derivatives(values, dt=1.0, order=2)

        scaled = coeffs.coefficients
        assert len(scaled) == 4
        # c_0 = f(9) = 81
        assert scaled[0] == pytest.approx(81.0, abs=1e-8)
        # c_1 = f'(9) = 17 (backward diff: 81-64 = 17)
        assert scaled[1] == pytest.approx(17.0, abs=1e-8)
        # c_2 = f''(9)/2 = 2/2 = 1
        assert scaled[2] == pytest.approx(1.0, abs=1e-8)
        # c_3 = f'''(9)/6 = 0 (order=2, so f'''=0)
        assert scaled[3] == pytest.approx(0.0, abs=1e-8)

    def test_to_dict_keys(self) -> None:
        coeffs = TaylorCoefficients(f_t=1.0, f_prime=2.0)
        d = coeffs.to_dict()
        assert set(d.keys()) == {"f_t", "f_prime", "f_double_prime", "f_triple_prime"}

    def test_estimated_order_tracks_data(self) -> None:
        """With only 2 points, order should be capped at 1."""
        coeffs = estimate_derivatives([5.0, 10.0], dt=1.0, order=3)
        assert coeffs.estimated_order == 1
        assert coeffs.f_prime == pytest.approx(5.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 3. TaylorDiagnostic structure
# ---------------------------------------------------------------------------

class TestTaylorDiagnostic:
    """Verify TaylorDiagnostic fields and serialization."""

    def test_linear_signal_stable(self) -> None:
        """Linear signal → stability_indicator = 0.0."""
        values = [float(i) for i in range(20)]
        coeffs = estimate_derivatives(values, dt=1.0, order=2)
        diag = compute_diagnostic(coeffs, values, dt=1.0)

        assert diag.stability_indicator == pytest.approx(0.0, abs=1e-10)
        assert diag.accel_variance == pytest.approx(0.0, abs=1e-10)
        assert diag.estimated_order == 2
        assert diag.local_slope == pytest.approx(1.0, abs=1e-10)
        assert diag.curvature == pytest.approx(0.0, abs=1e-10)

    def test_constant_signal_stable(self) -> None:
        values = [42.0] * 20
        coeffs = estimate_derivatives(values, dt=1.0, order=2)
        diag = compute_diagnostic(coeffs, values, dt=1.0)

        assert diag.stability_indicator == pytest.approx(0.0, abs=1e-10)
        assert diag.local_slope == pytest.approx(0.0, abs=1e-10)
        assert diag.curvature == pytest.approx(0.0, abs=1e-10)

    def test_quadratic_signal_stable(self) -> None:
        """Quadratic → constant f'' → accel_variance = 0."""
        values = [float(i * i) for i in range(20)]
        coeffs = estimate_derivatives(values, dt=1.0, order=2)
        diag = compute_diagnostic(coeffs, values, dt=1.0)

        assert diag.accel_variance == pytest.approx(0.0, abs=1e-8)
        assert diag.curvature == pytest.approx(2.0, abs=1e-8)

    def test_noisy_signal_unstable(self) -> None:
        """Alternating signal → high acceleration variance."""
        values = [10.0, 20.0, 10.0, 20.0, 10.0, 20.0, 10.0, 20.0]
        coeffs = estimate_derivatives(values, dt=1.0, order=2)
        diag = compute_diagnostic(coeffs, values, dt=1.0)

        assert diag.stability_indicator > 0.5
        assert diag.accel_variance > 0.0

    def test_to_dict_keys(self) -> None:
        values = [float(i) for i in range(20)]
        coeffs = estimate_derivatives(values, dt=1.0, order=2)
        diag = compute_diagnostic(coeffs, values, dt=1.0)
        d = diag.to_dict()

        expected_keys = {
            "estimated_order", "coefficients", "local_slope",
            "curvature", "stability_indicator", "accel_variance",
            "local_fit_error", "dt", "method",
        }
        assert set(d.keys()) == expected_keys

    def test_coefficients_length(self) -> None:
        values = [float(i) for i in range(20)]
        coeffs = estimate_derivatives(values, dt=1.0, order=2)
        diag = compute_diagnostic(coeffs, values, dt=1.0)

        assert len(diag.coefficients) == 4


# ---------------------------------------------------------------------------
# 4. Stability indicator
# ---------------------------------------------------------------------------

class TestStabilityIndicator:

    def test_zero_variance_zero_stability(self) -> None:
        assert compute_stability_indicator(0.0, 10.0) == 0.0

    def test_clamped_to_one(self) -> None:
        assert compute_stability_indicator(1e6, 1.0) == 1.0

    def test_near_zero_f_t_uses_unit_normalizer(self) -> None:
        result = compute_stability_indicator(0.5, 0.0)
        assert result == pytest.approx(0.5, abs=1e-10)


# ---------------------------------------------------------------------------
# 5. Non-uniform dt
# ---------------------------------------------------------------------------

class TestNonUniformDt:
    """Verify derivative accuracy degrades gracefully with non-uniform spacing."""

    def test_dt_scaling_linear(self) -> None:
        """f(t) = 2t with dt=0.5 → f' = 2."""
        values = [2.0 * 0.5 * i for i in range(20)]  # f(0), f(0.5), ..., f(9.5)
        coeffs = estimate_derivatives(values, dt=0.5, order=1)
        assert coeffs.f_prime == pytest.approx(2.0, abs=1e-10)

    def test_dt_from_timestamps(self) -> None:
        """Uniform timestamps with dt=2.0."""
        ts = [float(i * 2) for i in range(10)]
        dt = compute_dt(ts)
        assert dt == pytest.approx(2.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 6. Engine integration — diagnostic in metadata
# ---------------------------------------------------------------------------

class TestEngineEmitsDiagnostic:
    """Verify TaylorPredictionEngine includes diagnostic in metadata."""

    def test_diagnostic_present_in_metadata(self) -> None:
        from iot_machine_learning.infrastructure.ml.engines.taylor_engine import (
            TaylorPredictionEngine,
        )

        values = [float(i) for i in range(20)]
        engine = TaylorPredictionEngine(order=2, horizon=1)
        result = engine.predict(values)

        assert "diagnostic" in result.metadata
        diag = result.metadata["diagnostic"]
        assert "estimated_order" in diag
        assert "coefficients" in diag
        assert "local_slope" in diag
        assert "curvature" in diag
        assert "stability_indicator" in diag

    def test_fallback_has_null_diagnostic(self) -> None:
        from iot_machine_learning.infrastructure.ml.engines.taylor_engine import (
            TaylorPredictionEngine,
        )

        values = [10.0]
        engine = TaylorPredictionEngine(order=2, horizon=1)
        result = engine.predict(values)

        assert result.metadata["diagnostic"] is None
        assert result.metadata["fallback"] == "insufficient_data"

    def test_linear_signal_diagnostic_values(self) -> None:
        from iot_machine_learning.infrastructure.ml.engines.taylor_engine import (
            TaylorPredictionEngine,
        )

        values = [float(i) * 3.0 for i in range(20)]
        engine = TaylorPredictionEngine(order=2, horizon=1)
        result = engine.predict(values)

        diag = result.metadata["diagnostic"]
        assert diag["local_slope"] == pytest.approx(3.0, abs=1e-6)
        assert diag["curvature"] == pytest.approx(0.0, abs=1e-6)
        assert diag["stability_indicator"] == pytest.approx(0.0, abs=1e-6)
        assert diag["estimated_order"] == 2
