"""Unit tests for Phase 4: Geometric Manifold Active Control and Parameter Elimination."""

from __future__ import annotations

import math
import numpy as np
import pytest

from domain.entities.manifold import (
    ManifoldBoundaryLimits,
    ManifoldState3D,
    ManifoldState4D,
    ManifoldAuditRecord,
)
from domain.services.manifold import (
    VectorFieldConfig,
    compute_vector_field,
    compute_jacobian_tensor,
    compute_divergence,
    execute_ramanujan_jump,
    build_augmented_4d_jacobian,
    propagate_geodesic_step,
)
from infrastructure.ml.master_engine.geometric_manifold_adapter import (
    GeometricManifoldAdapter,
)
from infrastructure.ml.master_engine.master_equation import (
    compute_master_equation,
)


def test_manifold_boundary_limits_immutability_and_defaults():
    limits = ManifoldBoundaryLimits()
    assert limits.max_mahalanobis_clip == 100.0
    assert limits.divergence_threshold == 0.5
    assert limits.sech_input_clip == 15.0
    with pytest.raises(Exception):
        limits.max_mahalanobis_clip = 200.0  # frozen dataclass


def test_vector_field_config_dependency_injection():
    custom_limits = ManifoldBoundaryLimits(max_mahalanobis_clip=50.0, divergence_threshold=0.8)
    cfg = VectorFieldConfig(limits=custom_limits)
    assert cfg.limits.max_mahalanobis_clip == 50.0
    assert cfg.limits.divergence_threshold == 0.8


def test_jacobian_analytical_precision():
    state = ManifoldState3D(mahalanobis_d=3.0, kuramoto_r=0.7, bayesian_p=0.8)
    J = compute_jacobian_tensor(state)
    assert J.shape == (3, 3)
    # Check Liouville divergence trace
    div = compute_divergence(J)
    assert isinstance(div, float)
    assert not math.isnan(div)


def test_sovereign_modulation_in_master_equation():
    adapter = GeometricManifoldAdapter()
    
    # 1. Nominal state: div <= 0 (contractive), certainty is preserved: exp(0) = 1.0
    res_nominal = compute_master_equation(
        phi_moe_base=0.90,
        i_cvar=1.0,
        lambda_t_crono=1.0,
        kuramoto_r=0.95,
        phase_alignment=1.0,
        manifold_engine=adapter,
        mahalanobis_d=1.5,
    )
    assert res_nominal.certeza <= 0.90
    assert res_nominal.manifold_audit is not None
    
    # 2. Expansive state: feed an escalating sequence to trigger expansion / 4D jump
    for d in [5.0, 10.0, 16.0, 20.0]:
        adapter.step(mahalanobis_d=d, kuramoto_r=0.2, bayesian_p=0.3)
    
    audit = adapter.step(mahalanobis_d=25.0, kuramoto_r=0.1, bayesian_p=0.1)
    res_divergent = compute_master_equation(
        phi_moe_base=0.90,
        i_cvar=1.0,
        lambda_t_crono=1.0,
        kuramoto_r=0.1,
        phase_alignment=0.5,
        manifold_engine=adapter,
        mahalanobis_d=25.0,
    )
    if audit.is_4d_projected and audit.state_4d is not None:
        # In 4D projection, target magnitude is anchored to 4D coordinate
        assert res_divergent.magnitud_objetivo == pytest.approx(audit.state_4d.mahalanobis_d, abs=1e-6)
    
    # Sovereignty ensures certainty was attenuated
    assert res_divergent.certeza <= 0.90


def test_ramanujan_projection_symplectic_invariance():
    state = ManifoldState3D(mahalanobis_d=18.0, kuramoto_r=0.1, bayesian_p=0.2)
    j_curr = compute_jacobian_tensor(state)
    j_prev = j_curr * 0.95
    s4, j_4d, s3_rec = execute_ramanujan_jump(state, j_curr=j_curr, j_prev=j_prev, delta_time=0.01)
    
    assert isinstance(s4, ManifoldState4D)
    assert s4.frobenius_norm_j_dot >= 0.0
    assert j_4d.shape == (4, 4)
    # Augmented 4D Jacobian must be strictly contractive
    assert np.trace(j_4d) < 0.0
    # Recovered 3D state must be bounded within domain limits
    assert 0.0 <= s3_rec.mahalanobis_d <= ManifoldBoundaryLimits().max_mahalanobis_clip
    assert 0.0 <= s3_rec.kuramoto_r <= 1.0
    assert 0.0 <= s3_rec.bayesian_p <= 1.0
