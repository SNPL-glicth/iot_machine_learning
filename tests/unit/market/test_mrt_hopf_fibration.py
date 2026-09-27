"""Unit Tests for Hopf Spinor State and MRT Fibration Services (ZENIN v2.4+).

Verifies:
- ISO/IEC 25010: Stokes Pythagorean Invariant Conservation S₁² + S₂² + S₃² == S₀².
- Blind Spot 1: Dissipative Phase Transport bounded to [-π, π].
- Zero Transcendental: Exact precision of rational algebraic Hopf amplitudes.
- Four Sovereign Regimes: Quiescent, Chaotic Dissipation, Tesla Resonance, and Inverse Veto.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from domain.entities.manifold.hopf_spinor_state import HopfSpinorState
from domain.services.manifold.mrt_hopf_fibration import (
    compute_metric_deformation_rate,
    compute_rational_hopf_amplitudes,
    compute_vorticity_curl_norm,
    evaluate_hopf_spinor,
)
from domain.services.manifold.mrt_phase_transport import (
    reduce_angle_canonical,
    transport_phase_step,
)


def test_hopf_spinor_stokes_pythagorean_invariant() -> None:
    """Validate S₁² + S₂² + S₃² == S₀² within 1e-7 tolerance."""
    spinor = evaluate_hopf_spinor(
        nominal_certainty=0.85,
        frob_norm=12.5,
        phase_delta=0.45,
        trace_j4d_star=-0.5,
    )
    s0_sq = spinor.stokes_s0 ** 2
    s_vec_sq = (spinor.stokes_s1 ** 2) + (spinor.stokes_s2 ** 2) + (spinor.stokes_s3 ** 2)
    assert math.isclose(s_vec_sq, s0_sq, abs_tol=1e-7)
    assert 0.0 <= spinor.effective_certainty <= 1.0


def test_rational_hopf_amplitudes_match_exact_trigonometry() -> None:
    """Verify that rational algebra produces identical amplitudes to sin/cos/atan."""
    frob = 18.75
    c_nom = 0.9
    eps = 1e-3

    a1_rat, a2_rat, eta_rat = compute_rational_hopf_amplitudes(frob, c_nom, eps)

    # Reference trigonometric computation
    eta_ref = math.atan(frob / eps)
    a1_ref = math.sqrt(c_nom) * math.cos(eta_ref / 2.0)
    a2_ref = math.sqrt(c_nom) * math.sin(eta_ref / 2.0)

    assert math.isclose(a1_rat, a1_ref, rel_tol=1e-12)
    assert math.isclose(a2_rat, a2_ref, rel_tol=1e-12)
    assert math.isclose(eta_rat, eta_ref, rel_tol=1e-12)


def test_quiescent_regime_degenerates_to_nominal_consensus() -> None:
    """When ||J_dot||_F -> 0, C_sovereign must identically equal C_nominal."""
    c_nominal = 0.78
    spinor = evaluate_hopf_spinor(
        nominal_certainty=c_nominal,
        frob_norm=0.0,
        phase_delta=0.0,
    )
    assert math.isclose(spinor.stokes_s3, c_nominal, rel_tol=1e-9)
    assert math.isclose(spinor.stokes_s1, 0.0, abs_tol=1e-9)
    assert math.isclose(spinor.sovereign_certainty, c_nominal, rel_tol=1e-9)
    assert spinor.polarity_direction == 1.0


def test_tesla_resonance_recovers_certainty_under_extreme_shock() -> None:
    """Under extreme metric shock, in-phase alignment triggers Tesla resonance S₁."""
    c_nominal = 0.95
    spinor = evaluate_hopf_spinor(
        nominal_certainty=c_nominal,
        frob_norm=1e6,  # Extreme shock -> η -> π/2
        phase_delta=0.0,  # Perfect in-phase alignment -> cos(0) = 1
    )
    # Observable S3 collapses near 0
    assert abs(spinor.stokes_s3) < 1e-4
    # Resonance S1 absorbs full certainty
    assert math.isclose(spinor.stokes_s1, c_nominal, rel_tol=1e-4)
    assert math.isclose(spinor.sovereign_certainty, c_nominal, rel_tol=1e-4)
    assert spinor.is_tesla_resonant is True


def test_inverse_veto_generates_negative_polarity() -> None:
    """When anti-phase dominates, C_sovereign < 0 and polarity flips to -1.0."""
    spinor = evaluate_hopf_spinor(
        nominal_certainty=0.8,
        frob_norm=100.0,
        phase_delta=math.pi,  # Anti-phase cos(π) = -1.0
    )
    # S1 becomes strongly negative
    assert spinor.stokes_s1 < 0.0
    assert spinor.sovereign_certainty < 0.0
    assert spinor.polarity_direction == -1.0
    assert spinor.effective_certainty > 0.0


def test_dissipative_phase_transport_bounds_and_decays() -> None:
    """Ensure phase transport does not diverge over long time horizons."""
    phase = 0.0
    dt = 0.01
    for _ in range(500):
        phase, cos_p, sin_p = transport_phase_step(
            div_e=2.5,
            curl_b_norm=1.8,
            prev_phase_delta=phase,
            delta_time=dt,
            gamma_dissipation=0.2,
        )
        assert -math.pi <= phase <= math.pi
        assert math.isclose((cos_p ** 2) + (sin_p ** 2), 1.0, abs_tol=1e-9)


def test_vorticity_curl_extraction_from_jacobian() -> None:
    """Verify curl extraction from asymmetric components of Jacobian."""
    j = np.array([
        [0.0, -2.0, 1.0],
        [2.0, 0.0, -3.0],
        [-1.0, 3.0, 0.0],
    ], dtype=np.float64)
    # wx = -3 - 3 = -6, wy = 1 - (-1) = 2, wz = 2 - (-2) = 4
    # norm = sqrt(36 + 4 + 16) = sqrt(56) ≈ 7.4833
    curl_norm = compute_vorticity_curl_norm(j)
    assert math.isclose(curl_norm, math.sqrt(56.0), rel_tol=1e-9)


def test_adapter_produces_hopf_spinor_and_audit_trace() -> None:
    """Ensure GeometricManifoldAdapter injects HopfSpinorState into ManifoldAuditRecord."""
    from infrastructure.ml.master_engine.geometric_manifold_adapter import GeometricManifoldAdapter

    adapter = GeometricManifoldAdapter()
    audit1 = adapter.step(mahalanobis_d=2.0, kuramoto_r=0.8, bayesian_p=0.9, delta_time=0.01)
    assert audit1.hopf_spinor is not None
    assert audit1.hopf_spinor.stokes_s0 > 0.0

    trace = audit1.to_telemetry_trace()
    assert "hopf_spinor" in trace
    assert "stokes_s1_tesla" in trace["hopf_spinor"]
    assert "polarity_direction" in trace["hopf_spinor"]


def test_master_equation_hopf_integration_and_sovereignty() -> None:
    """Verify MasterEquation extracts sovereign components from Hopf manifold."""
    from infrastructure.ml.master_engine.geometric_manifold_adapter import GeometricManifoldAdapter
    from infrastructure.ml.master_engine.master_equation import compute_master_equation

    adapter = GeometricManifoldAdapter()
    for d in [2.0, 5.0, 12.0, 22.0]:
        adapter.step(mahalanobis_d=d, kuramoto_r=0.7, bayesian_p=0.8)

    res = compute_master_equation(
        phi_moe_base=0.85,
        i_cvar=1.0,
        lambda_t_crono=1.0,
        kuramoto_r=0.7,
        phase_alignment=0.9,
        manifold_engine=adapter,
        mahalanobis_d=25.0,
        manifold_shadow_mode=False,
    )
    assert res.hopf_spinor is not None
    assert res.sovereign_polarity in (-1.0, 1.0)
    assert res.geometric_manifold_shadow is not None
    assert "sovereign_certainty" in res.geometric_manifold_shadow
    assert res.certeza > 0.0

