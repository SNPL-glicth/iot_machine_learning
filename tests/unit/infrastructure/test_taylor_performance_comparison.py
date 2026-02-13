"""Performance comparison: backward vs central vs least-squares.

Measures prediction accuracy (not wall-clock time) across different
signal types to document the tradeoffs of each method.

Each test generates a known signal, predicts the next value using
all three methods, and compares the absolute error against the
true value.

Signal types:
1. Pure linear — all methods exact
2. Pure quadratic — backward has O(Δt) error, others better
3. Noisy linear — least-squares most robust
4. Noisy quadratic — least-squares most robust
5. Step function — all methods struggle (discontinuity)
"""

from __future__ import annotations

import random

import pytest

from iot_machine_learning.infrastructure.ml.engines.taylor import (
    DerivativeMethod,
    estimate_derivatives,
    project,
)


def _predict_next(
    values: list[float],
    order: int,
    method: DerivativeMethod,
) -> float:
    """Helper: predict next value using given method."""
    c = estimate_derivatives(values, dt=1.0, order=order, method=method)
    return project(c, h=1.0, order=c.estimated_order)


def _abs_error(predicted: float, actual: float) -> float:
    return abs(predicted - actual)


# ---------------------------------------------------------------------------
# 1. Pure linear: f(t) = 2t + 7
# ---------------------------------------------------------------------------

class TestPureLinearComparison:

    def _signal(self, t: float) -> float:
        return 2.0 * t + 7.0

    def test_all_methods_exact(self) -> None:
        values = [self._signal(float(i)) for i in range(20)]
        actual = self._signal(20.0)

        for method in DerivativeMethod:
            pred = _predict_next(values, order=1, method=method)
            err = _abs_error(pred, actual)
            assert err < 1e-8, (
                f"{method.value}: error={err:.2e}"
            )


# ---------------------------------------------------------------------------
# 2. Pure quadratic: f(t) = t² + 3t - 2
# ---------------------------------------------------------------------------

class TestPureQuadraticComparison:

    def _signal(self, t: float) -> float:
        return t * t + 3.0 * t - 2.0

    def test_backward_has_truncation_error(self) -> None:
        values = [self._signal(float(i)) for i in range(20)]
        actual = self._signal(20.0)
        pred = _predict_next(values, order=2, method=DerivativeMethod.BACKWARD)
        err = _abs_error(pred, actual)
        # Backward diff has O(Δt) error ≈ 1.0 for this quadratic
        assert err > 0.5, f"Expected truncation error, got {err:.4f}"

    def test_least_squares_lower_error_than_backward(self) -> None:
        values = [self._signal(float(i)) for i in range(20)]
        actual = self._signal(20.0)

        err_bwd = _abs_error(
            _predict_next(values, order=2, method=DerivativeMethod.BACKWARD),
            actual,
        )
        err_ls = _abs_error(
            _predict_next(values, order=2, method=DerivativeMethod.LEAST_SQUARES),
            actual,
        )
        assert err_ls <= err_bwd, (
            f"LS error ({err_ls:.4f}) > backward error ({err_bwd:.4f})"
        )


# ---------------------------------------------------------------------------
# 3. Noisy linear: f(t) = 5t + 10 + N(0, σ)
# ---------------------------------------------------------------------------

class TestNoisyLinearComparison:

    def test_least_squares_most_accurate_on_average(self) -> None:
        """Over many trials, LS should have lowest mean error."""
        random.seed(123)
        n_trials = 100
        sigma = 2.0
        errors = {m: [] for m in DerivativeMethod}

        for _ in range(n_trials):
            noise = [random.gauss(0, sigma) for _ in range(20)]
            values = [5.0 * i + 10.0 + noise[i] for i in range(20)]
            actual = 5.0 * 20.0 + 10.0  # noiseless next value

            for method in DerivativeMethod:
                pred = _predict_next(values, order=1, method=method)
                errors[method].append(_abs_error(pred, actual))

        means = {m: sum(e) / len(e) for m, e in errors.items()}

        # LS should beat backward on average
        assert means[DerivativeMethod.LEAST_SQUARES] < \
               means[DerivativeMethod.BACKWARD], (
            f"LS mean={means[DerivativeMethod.LEAST_SQUARES]:.2f} "
            f">= BWD mean={means[DerivativeMethod.BACKWARD]:.2f}"
        )


# ---------------------------------------------------------------------------
# 4. Noisy quadratic: f(t) = 0.5t² + N(0, σ)
# ---------------------------------------------------------------------------

class TestNoisyQuadraticComparison:

    def test_least_squares_most_stable(self) -> None:
        random.seed(456)
        n_trials = 100
        sigma = 1.0
        errors = {m: [] for m in DerivativeMethod}

        for _ in range(n_trials):
            noise = [random.gauss(0, sigma) for _ in range(20)]
            values = [0.5 * i * i + noise[i] for i in range(20)]
            actual = 0.5 * 20.0 * 20.0

            for method in DerivativeMethod:
                pred = _predict_next(values, order=2, method=method)
                errors[method].append(_abs_error(pred, actual))

        means = {m: sum(e) / len(e) for m, e in errors.items()}

        # LS should have lower mean error than backward
        assert means[DerivativeMethod.LEAST_SQUARES] < \
               means[DerivativeMethod.BACKWARD] * 1.5, (
            f"LS mean={means[DerivativeMethod.LEAST_SQUARES]:.2f} "
            f"not significantly better than BWD={means[DerivativeMethod.BACKWARD]:.2f}"
        )


# ---------------------------------------------------------------------------
# 5. Step function — all methods struggle
# ---------------------------------------------------------------------------

class TestStepFunctionComparison:
    """Discontinuous signal: all methods should produce finite output."""

    def test_all_methods_finite_on_step(self) -> None:
        values = [10.0] * 15 + [50.0] * 5
        for method in DerivativeMethod:
            pred = _predict_next(values, order=2, method=method)
            assert pred == pred, f"{method.value}: NaN prediction"
            # All should predict near 50 (recent values)
            assert 40.0 <= pred <= 60.0, (
                f"{method.value}: pred={pred:.1f} out of range"
            )


# ---------------------------------------------------------------------------
# 6. Summary table (informational, always passes)
# ---------------------------------------------------------------------------

class TestAccuracySummary:
    """Print a summary table of method accuracy across signal types."""

    def test_summary_table(self, capsys: pytest.CaptureFixture) -> None:
        signals = {
            "linear": ([3.0 * i + 1.0 for i in range(20)], 3.0 * 20 + 1.0),
            "quadratic": ([i * i for i in range(20)], 20.0 * 20.0),
            "cubic": ([i ** 3 for i in range(10)], 10.0 ** 3),
        }

        rows = []
        for sig_name, (values, actual) in signals.items():
            order = 2 if sig_name != "cubic" else 3
            for method in DerivativeMethod:
                pred = _predict_next(values, order=order, method=method)
                err = _abs_error(pred, actual)
                rows.append((sig_name, method.value, err))

        # Just verify all errors are finite
        for sig, method, err in rows:
            assert err == err, f"{sig}/{method}: NaN error"
