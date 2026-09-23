"""Unit tests for Sherman-Morrison incremental inverse covariance updates in MahalanobisFilter."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import inv

from infrastructure.ml.engines.rosa_roja.algorithms.modules.module1_ingestion import MahalanobisFilter


def test_sherman_morrison_numerical_exactness_vs_full_inversion():
    """Verify Sherman-Morrison rank-1 update tracks exact inverse covariance matrix."""
    dim = 4
    np.random.seed(42)
    mf = MahalanobisFilter(noise_threshold=5.0, history_window=200, min_samples_for_cov=5, recompute_interval=50)

    # Feed 45 observations (not hitting recompute_interval=50, testing pure Sherman-Morrison steps)
    for i in range(45):
        vec = np.random.randn(dim) * (1.0 + 0.1 * i)
        mf.process_raw_step(vec, 1.0)

        if mf._n >= 6:
            # Recompute exact regularized covariance
            cov = mf._M2 / (mf._n - 1)
            cov_reg = cov + mf.regularization_epsilon * np.eye(dim, dtype=np.float64)
            exact_inv = inv(cov_reg)
            exact_inv = 0.5 * (exact_inv + exact_inv.T)

            # Check that Sherman-Morrison tracked the exact inverse within numerical drift tolerance
            assert np.allclose(mf._cov_inv, exact_inv, atol=1e-3, rtol=1e-3), (
                f"Mismatch at step n={mf._n}:\nSherman:\n{mf._cov_inv}\nExact:\n{exact_inv}"
            )

            # Check strict symmetry
            assert np.allclose(mf._cov_inv, mf._cov_inv.T, atol=1e-12)


def test_sherman_morrison_active_symmetrization():
    """Verify cov_inv remains strictly symmetric across 150 iterations."""
    mf = MahalanobisFilter(noise_threshold=10.0, history_window=300, min_samples_for_cov=5, recompute_interval=50)
    np.random.seed(123)

    for _ in range(150):
        vec = np.random.randn(3)
        mf.process_raw_step(vec, 1.0)
        if mf._cov_inv is not None:
            diff = np.max(np.abs(mf._cov_inv - mf._cov_inv.T))
            assert diff == pytest.approx(0.0, abs=1e-14)


def test_sherman_morrison_collinear_safe_division():
    """Verify collinear vectors (rank deficient scatter) do not cause NaN or Inf."""
    mf = MahalanobisFilter(noise_threshold=10.0, history_window=50, min_samples_for_cov=5, division_epsilon=1e-6)
    base = np.array([2.0, -1.0, 0.5])

    for i in range(25):
        # Perfectly collinear vectors
        collinear_vec = base * (1.0 + i * 0.05)
        movement, is_outlier = mf.process_raw_step(collinear_vec, 1.0)
        assert movement is not None
        assert not np.isnan(movement.mahalanobis_distance)
        assert not np.isinf(movement.mahalanobis_distance)
        if mf._cov_inv is not None:
            assert not np.any(np.isnan(mf._cov_inv))
            assert not np.any(np.isinf(mf._cov_inv))


def test_periodic_recompute_anchor():
    """Verify that recompute_interval triggers exact reset of cov_inv."""
    interval = 10
    mf = MahalanobisFilter(noise_threshold=10.0, history_window=50, min_samples_for_cov=3, recompute_interval=interval)
    np.random.seed(99)

    for i in range(1, 25):
        mf.process_raw_step(np.random.randn(2), 1.0)
        if mf._n == interval or mf._n == 2 * interval:
            cov = mf._M2 / (mf._n - 1)
            cov_reg = cov + mf.regularization_epsilon * np.eye(2)
            expected = 0.5 * (inv(cov_reg) + inv(cov_reg).T)
            assert np.allclose(mf._cov_inv, expected, atol=1e-9)
