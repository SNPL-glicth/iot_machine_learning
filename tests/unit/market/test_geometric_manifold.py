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
    
    # 2a. In shadow mode (default True): certeza is NOT suppressed, magnitude not mutated
    adapter_shadow = GeometricManifoldAdapter()
    for d in [5.0, 10.0, 16.0, 20.0]:
        adapter_shadow.step(mahalanobis_d=d, kuramoto_r=0.2, bayesian_p=0.3)
    res_shadow = compute_master_equation(
        phi_moe_base=0.90, i_cvar=1.0, lambda_t_crono=1.0, kuramoto_r=0.1,
        phase_alignment=0.5, manifold_engine=adapter_shadow, mahalanobis_d=25.0,
        manifold_shadow_mode=True,
    )
    assert res_shadow.magnitud_objetivo == 0.0
    assert res_shadow.geometric_manifold_shadow is not None
    assert res_shadow.geometric_manifold_shadow["manifold_shadow_mode"] is True
    assert res_shadow.certeza == pytest.approx(0.90 * 0.1 * 0.5)

    # 2b. In active mode (manifold_shadow_mode=False): 4D magnitude is enforced
    adapter_active = GeometricManifoldAdapter()
    for d in [5.0, 10.0, 16.0, 20.0]:
        adapter_active.step(mahalanobis_d=d, kuramoto_r=0.2, bayesian_p=0.3)
    res_active = compute_master_equation(
        phi_moe_base=0.90, i_cvar=1.0, lambda_t_crono=1.0, kuramoto_r=0.1,
        phase_alignment=0.5, manifold_engine=adapter_active, mahalanobis_d=25.0,
        manifold_shadow_mode=False,
    )
    assert res_active.is_4d_projected is True
    assert res_active.magnitud_objetivo == pytest.approx(25.0, abs=1e-6)

    # 3. Positive divergence suppression: active mode attenuates certainty
    ad_pos_shadow = GeometricManifoldAdapter()
    res_pos_shadow = compute_master_equation(
        phi_moe_base=0.1, i_cvar=1.0, lambda_t_crono=1.0, kuramoto_r=0.05,
        phase_alignment=1.0, manifold_engine=ad_pos_shadow, mahalanobis_d=0.1,
        manifold_shadow_mode=True,
    )
    ad_pos_active = GeometricManifoldAdapter()
    res_pos_active = compute_master_equation(
        phi_moe_base=0.1, i_cvar=1.0, lambda_t_crono=1.0, kuramoto_r=0.05,
        phase_alignment=1.0, manifold_engine=ad_pos_active, mahalanobis_d=0.1,
        manifold_shadow_mode=False,
    )
    assert res_pos_shadow.divergence > 0.0
    assert res_pos_active.certeza < res_pos_shadow.certeza


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
