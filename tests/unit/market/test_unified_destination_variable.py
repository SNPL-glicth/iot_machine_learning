"""Unit tests for the Unified Destination Variable D(t) and Composite Pipeline.

Validates:
1. Closed Regression: Manifold Liouville suppression and MRT dual coupling execute
   in composition, NOT as mutually exclusive branches.
2. Exact 4-factor decomposition of D(t):
   D(t) = [∏_k I_k(t)] · exp(-max(0, div F(t))) · Pi_sov(t) · |S3(t) + S1(t)|
3. Dual-shadow mode preservation: D(t) is calculated in telemetry while final_certeza
   remains protected in nominal state.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock
import numpy as np
import pytest

from iot_machine_learning.infrastructure.ml.interfaces import PredictionResult
from iot_machine_learning.infrastructure.ml.master_engine import compute_master_equation
from iot_machine_learning.infrastructure.ml.master_engine.geometric_manifold_adapter import (
    GeometricManifoldAdapter,
)


def test_regression_bug_original_composed_liouville_and_mrt() -> None:
    """Confirms both MRT and Manifold execute together: Liouville suppresses dual coupling."""
    rr_res = PredictionResult(
        predicted_value=100.0, confidence=0.80, trend="up", metadata={"theta1_phase": 0.0},
    )
    mrt_res = PredictionResult(
        predicted_value=105.0, confidence=0.40, trend="up",
        metadata={"z2_amplitude": 0.40, "theta2_phase": 0.0},
    )

    # Manifold adapter with phi_moe_base=0.1, low kuramoto_r and low mahalanobis_d -> positive divergence Tr(J) > 0
    adapter_shadow = GeometricManifoldAdapter()
    res_shadow = compute_master_equation(
        phi_moe_base=0.10, i_cvar=1.0, lambda_t_crono=1.0, kuramoto_r=0.05,
        manifold_engine=adapter_shadow, mahalanobis_d=0.1,
        rosa_roja_output=rr_res, mrt_output=mrt_res,
        manifold_shadow_mode=True, dual_engine_shadow_mode=True,
    )
    assert res_shadow.geometric_manifold_shadow["dual_engine_mode"] is True
    assert res_shadow.geometric_manifold_shadow["manifold_shadow_mode"] is True
    assert res_shadow.divergence > 0.0
    # In shadow mode: final_certeza is unaffected nominal (0.10 * 0.05 = 0.005)
    assert res_shadow.certeza == pytest.approx(0.10 * 0.05)

    # 2. Active Mode: Liouville strictly dampens the coupled Hopf certainty!
    # In old bug (elif), this would fail because manifold was completely skipped.
    adapter_active = GeometricManifoldAdapter()
    res_active = compute_master_equation(
        phi_moe_base=0.10, i_cvar=1.0, lambda_t_crono=1.0, kuramoto_r=0.05,
        manifold_engine=adapter_active, mahalanobis_d=0.1,
        rosa_roja_output=rr_res, mrt_output=mrt_res,
        manifold_shadow_mode=False, dual_engine_shadow_mode=False,
    )
    assert res_active.divergence > 0.0
    expected_liouville = math.exp(-res_active.divergence)
    # Coupled certainty before Liouville: min(1.0, abs(sov_c))
    unsuppressed_c = min(1.0, abs(res_active.sovereign_certainty))
    expected_c = unsuppressed_c * expected_liouville
    assert res_active.certeza == pytest.approx(expected_c, rel=1e-5)
    assert res_active.certeza < unsuppressed_c


def test_variable_destino_four_factor_decomposition_exact() -> None:
    """Exact hand-calculated verification of D(t) = [∏ I_k] · exp(-max(0, div)) · Pi · |S3 + S1|."""
    # Case A: Constructive in-phase, positive divergence
    rr_res = PredictionResult(
        predicted_value=100.0, confidence=0.80, trend="up", metadata={"theta1_phase": 0.0},
    )
    mrt_res = PredictionResult(
        predicted_value=100.0, confidence=0.40, trend="up",
        metadata={"z2_amplitude": 0.40, "theta2_phase": 0.0},
    )

    mock_manifold = MagicMock()
    mock_audit = MagicMock()
    mock_audit.divergence = 0.25
    mock_audit.is_4d_projected = True
    mock_audit.hopf_spinor = None
    mock_audit.state_4d = None
    mock_manifold.step.return_value = mock_audit

    res_a = compute_master_equation(
        phi_moe_base=0.80, i_cvar=1.0, lambda_t_crono=1.0,
        manifold_engine=mock_manifold, mahalanobis_d=1.5,
        rosa_roja_output=rr_res, mrt_output=mrt_res,
        dual_engine_shadow_mode=True, manifold_shadow_mode=True,
    )

    # Hand calculation:
    # 1. ∏ I_k = I_mahal(1.5 <= 3.0 = 1) * I_takens(1.0) * I_risk(1.0 > 0 = 1) = 1.0
    # 2. Liouville: exp(-max(0, 0.25)) = exp(-0.25)
    # 3. S3 = 0.8^2 - 0.4^2 = 0.48; S1 = 2*0.8*0.4*cos(0) = 0.64; C_sov = 1.12
    # 4. Pi_sov = +1.0; |S3 + S1| = 1.12
    # D(t) = 1.0 * exp(-0.25) * (+1.0) * 1.12 = 0.87225687704
    expected_d_a = 1.0 * math.exp(-0.25) * 1.0 * 1.12
    assert res_a.variable_destino == pytest.approx(expected_d_a, rel=1e-6)
    assert res_a.D_t == pytest.approx(expected_d_a, rel=1e-6)

    # Case B: Destructive anti-phase (Veto Inverso), contractive divergence (div <= 0)
    rr_anti = PredictionResult(
        predicted_value=100.0, confidence=0.30, trend="up", metadata={"theta1_phase": 0.0},
    )
    mrt_anti = PredictionResult(
        predicted_value=85.0, confidence=0.90, trend="down",
        metadata={"z2_amplitude": 0.90, "theta2_phase": math.pi},
    )
    mock_audit.divergence = -0.50  # Contractive regime (div <= 0 -> exp(0) = 1.0)

    res_b = compute_master_equation(
        phi_moe_base=0.50, i_cvar=1.0, lambda_t_crono=1.0,
        manifold_engine=mock_manifold, mahalanobis_d=1.0,
        rosa_roja_output=rr_anti, mrt_output=mrt_anti,
    )

    # Hand calculation:
    # 1. ∏ I_k = 1.0
    # 2. Liouville: exp(-max(0, -0.50)) = exp(0) = 1.0
    # 3. S3 = 0.3^2 - 0.9^2 = -0.72; S1 = 2*0.3*0.9*cos(pi) = -0.54; C_sov = -1.26
    # 4. Pi_sov = -1.0; |S3 + S1| = 1.26
    # D(t) = 1.0 * 1.0 * (-1.0) * 1.26 = -1.26
    assert res_b.variable_destino == pytest.approx(-1.26, rel=1e-6)
    assert res_b.sovereign_polarity == -1.0


def test_variable_destino_admissibility_vetoes() -> None:
    """Verifies that Mahalanobis, Takens, or Risk vetoes zero out D(t) instantly."""
    rr = PredictionResult(predicted_value=100.0, confidence=0.7, trend="up", metadata={"theta1_phase": 0.0})
    mrt = PredictionResult(predicted_value=100.0, confidence=0.3, trend="up", metadata={"z2_amplitude": 0.3, "theta2_phase": 0.0})

    # 1. Mahalanobis contamination veto (d_M > 3.0) -> I_mahal = 0
    res_mahal_veto = compute_master_equation(
        phi_moe_base=0.7, i_cvar=1.0, lambda_t_crono=1.0, mahalanobis_d=4.5,
        rosa_roja_output=rr, mrt_output=mrt,
    )
    assert res_mahal_veto.i_admissibility == 0.0
    assert res_mahal_veto.variable_destino == 0.0

    # 2. Takens FNN topological veto -> I_takens = 0
    res_takens_veto = compute_master_equation(
        phi_moe_base=0.7, i_cvar=1.0, lambda_t_crono=1.0, mahalanobis_d=1.0,
        i_takens=0.0, rosa_roja_output=rr, mrt_output=mrt,
    )
    assert res_takens_veto.i_admissibility == 0.0
    assert res_takens_veto.variable_destino == 0.0

    # 3. Risk CVaR veto -> I_risk = 0
    res_risk_veto = compute_master_equation(
        phi_moe_base=0.7, i_cvar=0.0, lambda_t_crono=1.0, mahalanobis_d=1.0,
        rosa_roja_output=rr, mrt_output=mrt,
    )
    assert res_risk_veto.i_admissibility == 0.0
    assert res_risk_veto.variable_destino == 0.0


def test_combined_shadow_mode_isolation_and_telemetry() -> None:
    """With manifold_shadow_mode=True and dual_engine_shadow_mode=True, D(t) is recorded with zero interference."""
    rr = PredictionResult(predicted_value=100.0, confidence=0.75, trend="up", metadata={"theta1_phase": 0.0})
    mrt = PredictionResult(predicted_value=100.0, confidence=0.35, trend="up", metadata={"z2_amplitude": 0.35, "theta2_phase": 0.0})
    adapter = GeometricManifoldAdapter()
    res = compute_master_equation(
        phi_moe_base=0.75, i_cvar=1.0, lambda_t_crono=1.0, kuramoto_r=0.9, phase_alignment=1.0,
        manifold_engine=adapter, mahalanobis_d=1.0, rosa_roja_output=rr, mrt_output=mrt,
        manifold_shadow_mode=True, dual_engine_shadow_mode=True,
    )
    assert res.certeza == pytest.approx(0.75 * 0.9)
    assert res.magnitud_objetivo == 0.0
    assert res.variable_destino > 0.0
    assert res.geometric_manifold_shadow["variable_destino"] == res.variable_destino
    assert res.geometric_manifold_shadow["D_t"] == res.variable_destino
