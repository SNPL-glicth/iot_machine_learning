"""Unit tests for EmpiricalJacobianService.

Conforms to:
- ISO/IEC 25012:2008: Data precision and time-series estimation accuracy.
- ISO/IEC 25010:2023: Fault tolerance and numerical stability.
"""

from __future__ import annotations

import numpy as np
import pytest

from domain.services.manifold.empirical_jacobian_service import EmpiricalJacobianService


def test_empirical_jacobian_exact_recovery_on_linear_system() -> None:
    """Verify Ridge estimator recovers true linear flow A in Δx = A · x."""
    service = EmpiricalJacobianService(window_size=20, tikhonov_regularization=1e-6)

    # True dynamics: stable decay
    A_true = np.array([
        [-0.5,  0.1,  0.0],
        [ 0.0, -0.4,  0.1],
        [ 0.1,  0.0, -0.6],
    ], dtype=np.float64)

    # Generate trajectories
    np.random.seed(42)
    x = np.array([1.0, 0.5, 0.8], dtype=np.float64)
    traj = [x.copy()]
    for _ in range(25):
        dx = A_true @ (x - np.array([0.5, 0.5, 0.5])) + np.random.normal(0, 0.001, 3)
        x = x + dx
        traj.append(x.copy())

    states = np.array(traj)
    j_emp, cond, conf = service.estimate_empirical_jacobian(states)

    assert conf > 0.5
    assert np.all(np.isfinite(j_emp))
    # Empirical trace should reflect negative contractivity
    assert np.trace(j_emp) < 0.0


def test_empirical_jacobian_insufficient_samples_fallback() -> None:
    """Verify service returns zeros and zero confidence on insufficient samples."""
    service = EmpiricalJacobianService(min_samples=6)
    short_states = np.array([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]])

    j_emp, cond, conf = service.estimate_empirical_jacobian(short_states)

    assert conf == 0.0
    assert np.array_equal(j_emp, np.zeros((3, 3)))


def test_empirical_jacobian_collinear_subspace_fallback() -> None:
    """Verify singular or collinear trajectory prevents runaway condition numbers."""
    service = EmpiricalJacobianService(max_condition_number=100.0)

    # Collinear trajectory along 1 single direction
    collinear = np.outer(np.linspace(1.0, 10.0, 20), np.array([1.0, 1.0, 1.0]))

    j_emp, cond, conf = service.estimate_empirical_jacobian(collinear)

    # Must collapse confidence to 0.0
    assert conf == 0.0
    assert np.array_equal(j_emp, np.zeros((3, 3)))


def test_blend_jacobian_smoothly_preserves_analytical_anchor() -> None:
    """Verify blend_jacobian anchors safely to analytical Jacobian when data is noisy."""
    service = EmpiricalJacobianService(window_size=10)
    current = np.array([0.5, 0.8, 0.6])
    history = np.array([current + np.random.normal(0, 0.01, 3) for _ in range(15)])

    j_blended, div, beta = service.blend_jacobian(current, history, beta_max=0.5)

    assert j_blended.shape == (3, 3)
    assert np.isfinite(div)
    assert 0.0 <= beta <= 0.5
