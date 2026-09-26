"""Unit tests for Phase 2 Takens Mathematical Domain Services."""

from __future__ import annotations

import numpy as np
import pytest

from domain.entities.takens import TakensParameters, TopologicalAuditRecord
from domain.services.takens import (
    extract_embedded_vector,
    extract_delay_matrix,
    compute_delay_covariance_matrix,
    compute_effective_dimension,
    compute_spectral_entropy,
    compute_fast_fnn_ratio,
    evaluate_topological_fidelity,
)


def test_delay_embedding_extraction():
    params = TakensParameters(m=4, tau_strides=(1, 2, 4))
    # Synthetic series 0, 1, 2, ..., 19
    series = np.arange(20, dtype=np.float64)

    # Expected vector: [x_19, x_18, x_17, x_15]
    vec = extract_embedded_vector(series, params)
    assert vec.shape == (4,)
    assert vec[0] == 19.0
    assert vec[1] == 18.0
    assert vec[2] == 17.0
    assert vec[3] == 15.0

    # Test short series with NaNs
    short = np.array([5.0, np.nan], dtype=np.float64)
    vec_short = extract_embedded_vector(short, params)
    assert vec_short.shape == (4,)
    assert np.all(np.isfinite(vec_short))


def test_delay_matrix_extraction():
    params = TakensParameters(m=3, tau_strides=(1, 2))
    series = np.arange(10, dtype=np.float64)
    matrix = extract_delay_matrix(series, window_size=5, params=params)

    assert matrix.shape == (5, 3)
    # The last row should be the most recent embedded vector: [9, 8, 7]
    assert np.allclose(matrix[-1], [9.0, 8.0, 7.0])
    # The previous row should be: [8, 7, 6]
    assert np.allclose(matrix[-2], [8.0, 7.0, 6.0])


def test_spectral_covariance_and_effective_dimension():
    # 1. Collinear rank-1 signal: effective dimension must approach 1.0
    t = np.linspace(0, 10, 100)
    collinear_matrix = np.column_stack([t, 2 * t, 3 * t, 4 * t])
    cov_rank1 = compute_delay_covariance_matrix(collinear_matrix, tikhonov_reg=1e-8)
    assert cov_rank1.shape == (4, 4)
    assert np.allclose(cov_rank1, cov_rank1.T)  # Symmetry

    d_eff_rank1 = compute_effective_dimension(cov_rank1, m_dimension=4)
    assert 1.0 <= d_eff_rank1 <= 1.05  # Highly concentrated in 1st component

    # 2. Isotropic white noise in R^4: effective dimension should approach 4.0
    rng = np.random.default_rng(42)
    noise_matrix = rng.normal(0, 1, size=(500, 4))
    cov_noise = compute_delay_covariance_matrix(noise_matrix, tikhonov_reg=1e-6)
    d_eff_noise = compute_effective_dimension(cov_noise, m_dimension=4)
    assert 3.7 <= d_eff_noise <= 4.0

    # 3. Spectral entropy
    _, entropy = compute_spectral_entropy(cov_noise)
    assert entropy > 0.0


def test_topological_fidelity_and_veto():
    params = TakensParameters(
        m=3,
        tau_strides=(1, 2),
        tau_fnn=0.40,
        max_geodesic_distance=5.0,
        spectral_collapse_threshold=1.10,
    )

    # Reconstructed history: stationary orbit around (10, 10, 10)
    rng = np.random.default_rng(123)
    historic_matrix = 10.0 + rng.normal(0, 0.1, size=(50, 3))

    # Case 1: Nominal trajectory in proximity to centroid
    traj_nominal = [9.9, 10.0, 10.1, 10.05]
    psi, audit = evaluate_topological_fidelity(
        trajectory_values=traj_nominal,
        historic_matrix=historic_matrix,
        effective_dimension=2.5,
        params=params,
        timestamp_ns=1000,
    )
    assert 0.7 <= psi <= 1.0
    assert audit.is_manifold_veto is False
    assert audit.veto_reason is None

    # Case 2: Out-of-manifold escape (huge distance from centroid)
    traj_escape = [100.0, 105.0, 110.0]
    psi_esc, audit_esc = evaluate_topological_fidelity(
        trajectory_values=traj_escape,
        historic_matrix=historic_matrix,
        effective_dimension=2.5,
        params=params,
        timestamp_ns=2000,
    )
    assert psi_esc == 0.0
    assert audit_esc.is_manifold_veto is True
    assert "out_of_manifold_escape" in audit_esc.veto_reason

    # Case 3: Spectral collapse veto
    psi_collapse, audit_collapse = evaluate_topological_fidelity(
        trajectory_values=traj_nominal,
        historic_matrix=historic_matrix,
        effective_dimension=1.05,  # Below threshold 1.10
        params=params,
        timestamp_ns=3000,
    )
    assert psi_collapse == 0.0
    assert audit_collapse.is_manifold_veto is True
    assert "spectral_collapse" in audit_collapse.veto_reason
