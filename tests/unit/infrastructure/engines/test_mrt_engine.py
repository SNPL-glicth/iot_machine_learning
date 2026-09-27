"""Unit Tests for MRTEngine and PhaseConjugator Dual-Engine Integration (ZENIN v2.4+).

Verifies:
- MRTEngine conforms to PredictionEngine interface.
- PhaseConjugator computes vorticity and dissipative phase transport with zero transcendentals.
- MasterEquation orchestrator unites Rosa Roja (z1) and MRT (z2) via Hopf Fibration.
- Inverse Kinematics rebound vector and Veto Inverso execution.
"""

from __future__ import annotations

import math
import pytest

from iot_machine_learning.infrastructure.ml.engines.mrt.mrt_engine import MRTEngine
from iot_machine_learning.infrastructure.ml.engines.mrt.phase_conjugator import PhaseConjugator
from iot_machine_learning.infrastructure.ml.interfaces import PredictionResult
from iot_machine_learning.infrastructure.ml.master_engine.master_equation import compute_master_equation


def test_mrt_engine_predict_elastic_rebound() -> None:
    """MRT evaluates elastic rebound trajectory opposite to acute drop (Flash Crash)."""
    engine = MRTEngine(elasticity_rebound=1.0)
    # Steep drop values: 100 -> 95 -> 88 -> 78 -> 65
    values = [100.0, 95.0, 88.0, 78.0, 65.0]
    timestamps = [1.0, 2.0, 3.0, 4.0, 5.0]

    result = engine.predict(values, timestamps)

    assert isinstance(result, PredictionResult)
    # Forward velocity is negative (drop), so elastic rebound should be positive (up)
    assert result.predicted_value > 65.0
    assert result.trend == "up"
    assert 0.0 <= result.confidence <= 1.0
    assert result.metadata["engine_name"] == "mrt"
    assert result.metadata["kinematics_mode"] == "inverse_conjugate_mirror"
    assert result.metadata["z2_amplitude"] >= 0.0


def test_mrt_engine_insufficient_history_fallback() -> None:
    """MRT returns fallback prediction when observation window is too short."""
    engine = MRTEngine(min_history_points=5)
    result = engine.predict([10.0, 11.0])

    assert result.confidence == 0.0
    assert result.trend == "stable"
    assert result.metadata["status"] == "insufficient_history"


def test_phase_conjugator_rational_vorticity_and_phase() -> None:
    """PhaseConjugator computes bounded phase in [-π, π] and positive vorticity."""
    conjugator = PhaseConjugator()
    values = [10.0, 12.0, 15.0, 11.0, 7.0, 4.0]
    sin_half, theta2, curl_b, frob_rate, vel = conjugator.compute_phase_conjugation(values, delta_time=1.0)

    assert 0.0 <= sin_half <= 1.0
    assert -math.pi <= theta2 <= math.pi
    assert curl_b >= 0.0
    assert frob_rate >= 0.0
    assert vel == pytest.approx(-3.0)


def test_master_equation_dual_engine_hopf_coupling() -> None:
    """Master Equation couples Rosa Roja (z1) and MRT (z2) via Hopf Fibration."""
    # Simulated Rosa Roja Output (Forward Kinematics)
    rosa_roja_res = PredictionResult(
        predicted_value=105.0,
        confidence=0.85,
        trend="up",
        metadata={"theta1_phase": 0.0},
    )

    # Simulated MRT Output (Inverse Kinematics / Elastic Rebound)
    mrt_res = PredictionResult(
        predicted_value=110.0,
        confidence=0.40,
        trend="up",
        metadata={"z2_amplitude": 0.40, "theta2_phase": 0.0},
    )

    # In-phase coupling (θ1 - θ2 = 0) -> Constructive Tesla Resonance
    res = compute_master_equation(
        phi_moe_base=0.85,
        i_cvar=1.0,
        lambda_t_crono=1.0,
        rosa_roja_output=rosa_roja_res,
        mrt_output=mrt_res,
        dual_engine_shadow_mode=False,
    )

    assert res.geometric_manifold_shadow is not None
    assert res.geometric_manifold_shadow["dual_engine_mode"] is True
    # C_sovereign = |z1|² - |z2|² + 2|z1z2|cos(0) = (0.85² - 0.40²) + 2*(0.85*0.40)*1.0
    expected_s3 = 0.85**2 - 0.40**2
    expected_s1 = 2.0 * 0.85 * 0.40 * 1.0
    expected_c = expected_s3 + expected_s1
    assert math.isclose(res.sovereign_certainty, expected_c, rel_tol=1e-5)
    assert res.sovereign_polarity == 1.0
    assert math.isclose(res.certeza, min(1.0, expected_c), rel_tol=1e-5)
    # Target magnitude blends both engines
    assert 105.0 <= res.magnitud_objetivo <= 110.0


def test_master_equation_dual_engine_veto_inverso() -> None:
    """Master Equation executes Veto Inverso when phase is in destructive opposition."""
    # Rosa Roja predicts continuation up
    rr_res = PredictionResult(
        predicted_value=100.0,
        confidence=0.30,
        trend="up",
        metadata={"theta1_phase": 0.0},
    )

    # MRT sees violent vortex in 4D (high z2) and anti-phase (θ2 = π)
    mrt_res = PredictionResult(
        predicted_value=85.0,
        confidence=0.90,
        trend="down",
        metadata={"z2_amplitude": 0.90, "theta2_phase": math.pi},
    )

    res = compute_master_equation(
        phi_moe_base=0.5,
        i_cvar=1.0,
        lambda_t_crono=1.0,
        rosa_roja_output=rr_res,
        mrt_output=mrt_res,
        dual_engine_shadow_mode=False,
    )

    # C_sovereign = (0.3² - 0.9²) + 2*(0.3*0.9)*cos(-π) = (0.09 - 0.81) - 0.54 = -1.26 < 0
    assert res.sovereign_certainty < 0.0
    assert res.sovereign_polarity == -1.0
    # Shielded target magnitude: polarity modulates displacement delta_y, protecting absolute price
    assert res.magnitud_objetivo > 0.0
    assert res.delta_y_sovereign > 0.0
    assert res.magnitud_objetivo == pytest.approx(113.5)


def test_master_equation_orchestrator_parallel_mrt_execution() -> None:
    """MasterEquationOrchestrator executes Rosa Roja and MRT in parallel."""
    from unittest.mock import MagicMock
    import numpy as np
    from domain.entities.rosa_roja.execution import ExecutionPlan
    from infrastructure.ml.master_engine.orchestrator import MasterEquationOrchestrator

    mock_rr, mock_risk, mock_temporal = MagicMock(), MagicMock(), MagicMock()
    mock_rr.process_event.return_value = ExecutionPlan.HOLD(reason="test")
    mock_risk.record_observation.return_value = {"veto_riesgo": 1.0}
    mock_temporal.record_observation.return_value = {"lambda_crono": 1.0, "dS_dt": 0.01, "dR_dt": 0.01}

    orch = MasterEquationOrchestrator(
        rosa_roja_engine=mock_rr,
        risk_adapter=mock_risk,
        temporal_adapter=mock_temporal,
        mrt_engine=MRTEngine(),
        shadow_mode=False,
    )

    delta = np.array([10.0, 12.0, 15.0, 11.0, 8.0], dtype=np.float64)
    plan = orch.process_event(delta, 1.0)

    assert isinstance(plan, ExecutionPlan)
    trace = plan.envelope.decision_trace if plan.envelope else plan.veto_details.get("decision_trace", {})
    shadow = trace.get("geometric_manifold_shadow", {})
    assert shadow.get("dual_engine_mode") is True
    assert shadow.get("z2_amplitude", 0.0) >= 0.0
    assert shadow.get("protected_target_magnitude", 0.0) > 0.0

