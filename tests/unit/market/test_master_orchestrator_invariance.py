"""Invariance test suite for MasterEquationOrchestrator.

Validates that over >= 30 simulated market cycles:
1. Orchestrator in shadow mode produces 100% identical actions (EXECUTE/HOLD/EMERGENCY_FLUSH)
   as pure RosaRojaEngine for the identical input sequence.
2. Global confidence matches phi_moe_base strictly.
3. Master Equation (phi_redrose = I_cvar * lambda_crono * phi_moe_base) is computed and
   recorded into ISO 22989 decision_trace on every cycle.
"""

from __future__ import annotations

import numpy as np
import pytest

from iot_machine_learning.infrastructure.ml.adapters import (
    KalmanExpertAdapter,
    RiskEngineAdapter,
    StatisticalExpertAdapter,
    TaylorExpertAdapter,
    TemporalEngineAdapter,
)
from iot_machine_learning.infrastructure.ml.engines.kalman.engine import KalmanPredictionEngine
from iot_machine_learning.infrastructure.ml.engines.rosa_roja.algorithms.engine import RosaRojaEngine
from iot_machine_learning.infrastructure.ml.engines.rosa_roja.algorithms.modules.module1_ingestion import (
    MahalanobisFilter,
)
from iot_machine_learning.infrastructure.ml.engines.rosa_roja.algorithms.modules.module3_moe_gating import (
    MultiplicativeMoEGating,
)
from iot_machine_learning.infrastructure.ml.engines.rosa_roja.algorithms.modules.rhythm_generator import (
    RhythmTrajectoryGenerator,
)
from iot_machine_learning.infrastructure.ml.engines.statistical import StatisticalPredictionEngine
from iot_machine_learning.infrastructure.ml.engines.taylor.engine import TaylorPredictionEngine
from iot_machine_learning.infrastructure.ml.master_engine import (
    MasterEquationOrchestrator,
    compute_master_equation,
)


def _build_rosa_roja_engine() -> RosaRojaEngine:
    ingestion = MahalanobisFilter(noise_threshold=3.0, history_window=50, min_samples_for_cov=10)
    rhythm = RhythmTrajectoryGenerator(min_trajectory_len=11, max_trajectory_len=15, top_k=4)
    gating = MultiplicativeMoEGating(variance_penalty=0.5)
    jury = [
        TaylorExpertAdapter(engine=TaylorPredictionEngine()),
        KalmanExpertAdapter(engine=KalmanPredictionEngine()),
        StatisticalExpertAdapter(engine=StatisticalPredictionEngine()),
    ]
    return RosaRojaEngine(
        ingestion_filter=ingestion,
        rhythm_generator=rhythm,
        moe_gating=gating,
        expert_jury=jury,
        drift_sensors=[],
    )


class TestMasterOrchestratorInvariance:
    """Verifies that MasterEquationOrchestrator adheres strictly to shadow mode non-interference."""

    def test_thirty_cycles_invariance_and_master_equation_trace(self) -> None:
        """Runs 35 cycles comparing direct Rosa Roja vs MasterEquationOrchestrator."""
        engine_base = _build_rosa_roja_engine()
        engine_for_orch = _build_rosa_roja_engine()

        risk_adapter = RiskEngineAdapter(l_max=0.03, default_sigma=0.002)
        temporal_adapter = TemporalEngineAdapter()

        orchestrator = MasterEquationOrchestrator(
            rosa_roja_engine=engine_for_orch,
            risk_adapter=risk_adapter,
            temporal_adapter=temporal_adapter,
            shadow_mode=True,
        )

        rng = np.random.default_rng(42)
        num_cycles = 35

        for step in range(num_cycles):
            delta = rng.normal(0, 0.0015, size=10)
            delta[0] = 0.004  # Stable upward drift to allow valid candidate trajectories
            delta[5] = float(rng.uniform(-0.5, 0.5))  # Order book imbalance signal
            delta_time = 1.0

            plan_base = engine_base.process_event(delta, delta_time)
            plan_orchestrator = orchestrator.process_event(delta, delta_time)

            # Invariance Check 1: Action (EXECUTE / HOLD / EMERGENCY_FLUSH) must be identical
            assert plan_orchestrator.action == plan_base.action, (
                f"Cycle {step}: Action mismatch {plan_orchestrator.action} != {plan_base.action}"
            )

            # Invariance Check 2: Confidence must match Rosa Roja phi_moe_base in shadow mode
            assert plan_orchestrator.global_confidence == pytest.approx(
                plan_base.global_confidence, abs=1e-12
            ), f"Cycle {step}: Confidence divergence"

            # Invariance Check 3: Extract decision_trace and verify Master Equation terms
            trace = None
            if plan_orchestrator.envelope and plan_orchestrator.envelope.metadata:
                trace = plan_orchestrator.envelope.metadata.get("decision_trace")
            elif getattr(plan_orchestrator, "veto_details", None) and isinstance(plan_orchestrator.veto_details, dict):
                trace = plan_orchestrator.veto_details.get("decision_trace")

            assert trace is not None, f"Cycle {step}: decision_trace missing in plan"
            assert "phi_moe_base" in trace
            assert "I_cvar" in trace
            assert "lambda_t_crono" in trace
            assert "phi_redrose" in trace
            assert trace["governing_component"] == "phi_moe_base"
            assert trace["execution_mode"] == "shadow"

            # Invariance Check 4: Exact mathematical consistency of Master Equation
            phi_base = trace["phi_moe_base"]
            i_cvar = trace["I_cvar"]
            lambda_crono = trace["lambda_t_crono"]
            phi_redrose = trace["phi_redrose"]

            expected_redrose = i_cvar * lambda_crono * phi_base
            assert phi_redrose == pytest.approx(expected_redrose, abs=1e-9), (
                f"Cycle {step}: Formula mismatch: {phi_redrose} != {i_cvar} * {lambda_crono} * {phi_base}"
            )

    def test_numerical_example_evaluation(self) -> None:
        """Evaluates concrete numerical example to ensure exact step-by-step arithmetic."""
        res = compute_master_equation(
            phi_moe_base=0.85,
            i_cvar=1.0,
            lambda_t_crono=0.90,
        )
        assert res.phi_moe_base == 0.85
        assert res.i_cvar == 1.0
        assert res.lambda_t_crono == 0.90
        # 1.0 * 0.90 * 0.85 = 0.765
        assert res.phi_redrose == pytest.approx(0.765, abs=1e-9)

        # Risk veto scenario: CVaR breach sets I_cvar = 0.0
        res_veto = compute_master_equation(
            phi_moe_base=0.85,
            i_cvar=0.0,
            lambda_t_crono=0.90,
        )
        assert res_veto.i_cvar == 0.0
        assert res_veto.phi_redrose == 0.0
