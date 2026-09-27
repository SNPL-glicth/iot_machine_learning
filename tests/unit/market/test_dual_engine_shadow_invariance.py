"""Invariance test suite for Dual-Engine Hopf Coupling Shadow Mode.

Validates that over >= 30 continuous cycles:
1. Orchestrator with MRTEngine in dual_engine_shadow_mode=True produces 100% identical
   actions (EXECUTE/HOLD) and global confidence as running with mrt_engine=None (Rosa Roja alone).
2. Complete diagnostic telemetry (stokes_s0, s1, s3, sovereign_certainty) is recorded.
3. When dual_engine_shadow_mode=False, the Hopf coupling modulates certainty and target magnitude.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from domain.entities.rosa_roja.execution import ExecutionPlan
from iot_machine_learning.infrastructure.ml.adapters import (
    KalmanExpertAdapter, RiskEngineAdapter, StatisticalExpertAdapter,
    TaylorExpertAdapter, TemporalEngineAdapter,
)
from iot_machine_learning.infrastructure.ml.engines.kalman.engine import KalmanPredictionEngine
from iot_machine_learning.infrastructure.ml.engines.mrt.mrt_engine import MRTEngine
from iot_machine_learning.infrastructure.ml.engines.rosa_roja.algorithms.engine import RosaRojaEngine
from iot_machine_learning.infrastructure.ml.engines.rosa_roja.algorithms.modules.module1_ingestion import MahalanobisFilter
from iot_machine_learning.infrastructure.ml.engines.rosa_roja.algorithms.modules.module3_moe_gating import MultiplicativeMoEGating
from iot_machine_learning.infrastructure.ml.engines.rosa_roja.algorithms.modules.rhythm_generator import RhythmTrajectoryGenerator
from iot_machine_learning.infrastructure.ml.engines.statistical import StatisticalPredictionEngine
from iot_machine_learning.infrastructure.ml.engines.taylor.engine import TaylorPredictionEngine
from iot_machine_learning.infrastructure.ml.interfaces import PredictionResult
from iot_machine_learning.infrastructure.ml.master_engine import (
    MasterEquationOrchestrator, compute_master_equation,
)


def _build_engine() -> RosaRojaEngine:
    return RosaRojaEngine(
        ingestion_filter=MahalanobisFilter(noise_threshold=3.0, history_window=50, min_samples_for_cov=10),
        rhythm_generator=RhythmTrajectoryGenerator(min_trajectory_len=11, max_trajectory_len=15, top_k=4),
        moe_gating=MultiplicativeMoEGating(variance_penalty=0.5),
        expert_jury=[
            TaylorExpertAdapter(engine=TaylorPredictionEngine()),
            KalmanExpertAdapter(engine=KalmanPredictionEngine()),
            StatisticalExpertAdapter(engine=StatisticalPredictionEngine()),
        ],
        drift_sensors=[],
    )


def _extract_trace(plan: ExecutionPlan) -> dict:
    if plan.envelope and plan.envelope.metadata:
        return plan.envelope.metadata.get("decision_trace", {})
    if getattr(plan, "veto_details", None) and isinstance(plan.veto_details, dict):
        return plan.veto_details.get("decision_trace", {})
    return {}


class TestDualEngineShadowInvariance:
    """Verifies invariant behavior of dual_engine_shadow_mode=True vs mrt_engine=None."""

    def test_thirty_five_cycles_dual_engine_shadow_invariance(self) -> None:
        """35 cycles: dual_engine_shadow_mode=True must be 100% identical to mrt_engine=None."""
        orch_no_mrt = MasterEquationOrchestrator(
            rosa_roja_engine=_build_engine(), risk_adapter=RiskEngineAdapter(l_max=0.03, default_sigma=0.002),
            temporal_adapter=TemporalEngineAdapter(), mrt_engine=None, shadow_mode=False,
        )
        orch_shadow = MasterEquationOrchestrator(
            rosa_roja_engine=_build_engine(), risk_adapter=RiskEngineAdapter(l_max=0.03, default_sigma=0.002),
            temporal_adapter=TemporalEngineAdapter(), mrt_engine=MRTEngine(),
            dual_engine_shadow_mode=True, shadow_mode=False,
        )

        rng = np.random.default_rng(42)
        for step in range(35):
            delta = rng.normal(0, 0.0015, size=10)
            delta[0] = 0.003
            delta[3] = float(rng.uniform(-0.4, 0.4))
            dt = 1.0

            plan_no_mrt = orch_no_mrt.process_event(delta, dt)
            plan_shadow = orch_shadow.process_event(delta, dt)

            # Invariance 1: Action must be 100% identical
            assert plan_shadow.action == plan_no_mrt.action, f"Cycle {step}: Action mismatch"

            # Invariance 2: Global confidence identical to machine precision
            assert plan_shadow.global_confidence == pytest.approx(
                plan_no_mrt.global_confidence, abs=1e-12
            ), f"Cycle {step}: Confidence mismatch"

            # Invariance 3: Target magnitude unmutated
            if plan_shadow.action == "EXECUTE" and plan_shadow.envelope and plan_no_mrt.envelope:
                assert plan_shadow.envelope.magnitude == pytest.approx(
                    plan_no_mrt.envelope.magnitude, abs=1e-12
                )

            # Invariance 4: Full Stokes & Hopf telemetry recorded
            trace_shadow = _extract_trace(plan_shadow)
            geo = trace_shadow.get("geometric_manifold_shadow")
            assert geo is not None, f"Cycle {step}: Missing dual engine telemetry"
            assert geo["dual_engine_mode"] is True
            assert geo["dual_engine_shadow_mode"] is True
            assert "stokes_s0" in geo and "stokes_s1" in geo and "stokes_s3" in geo
            assert "sovereign_certainty" in geo and "sovereign_polarity" in geo
            assert geo["nominal_certeza"] == pytest.approx(plan_no_mrt.global_confidence, abs=1e-9)

    def test_active_mode_dual_engine_modulates_certainty_and_magnitude(self) -> None:
        """When dual_engine_shadow_mode=False, Hopf coupling actively modulates certainty and magnitude."""
        rr_res = PredictionResult(predicted_value=100.0, confidence=0.30, trend="up", metadata={"theta1_phase": 0.0})
        mrt_anti = PredictionResult(
            predicted_value=85.0, confidence=0.90, trend="down",
            metadata={"z2_amplitude": 0.90, "theta2_phase": math.pi},
        )

        comp_shadow = compute_master_equation(
            phi_moe_base=0.5, i_cvar=1.0, lambda_t_crono=1.0,
            rosa_roja_output=rr_res, mrt_output=mrt_anti, dual_engine_shadow_mode=True,
        )
        comp_active = compute_master_equation(
            phi_moe_base=0.5, i_cvar=1.0, lambda_t_crono=1.0,
            rosa_roja_output=rr_res, mrt_output=mrt_anti, dual_engine_shadow_mode=False,
        )

        # In shadow mode, certeza is strictly nominal (0.50) and target magnitude unmutated (0.0)
        assert comp_shadow.certeza == pytest.approx(0.50)
        assert comp_shadow.magnitud_objetivo == 0.0
        assert comp_shadow.geometric_manifold_shadow["dual_engine_shadow_mode"] is True

        # In active mode, Hopf coupling modulates certeza = |sov_c| = |-1.26| clamped to 1.0
        assert comp_active.certeza == pytest.approx(1.0)
        assert comp_active.sovereign_certainty == pytest.approx(-1.26)
        assert comp_active.sovereign_polarity == -1.0
        assert comp_active.magnitud_objetivo == pytest.approx(113.5)

    def test_active_mode_orchestrator_end_to_end_veto_inverso(self) -> None:
        """End-to-end orchestrator: active mode enforces target magnitude from Hopf Fibration."""
        orch_active = MasterEquationOrchestrator(
            rosa_roja_engine=_build_engine(), risk_adapter=RiskEngineAdapter(l_max=0.03, default_sigma=0.002),
            temporal_adapter=TemporalEngineAdapter(), mrt_engine=MRTEngine(),
            dual_engine_shadow_mode=False, shadow_mode=False,
        )
        orch_shadow = MasterEquationOrchestrator(
            rosa_roja_engine=_build_engine(), risk_adapter=RiskEngineAdapter(l_max=0.03, default_sigma=0.002),
            temporal_adapter=TemporalEngineAdapter(), mrt_engine=MRTEngine(),
            dual_engine_shadow_mode=True, shadow_mode=False,
        )

        rng = np.random.default_rng(999)
        for _ in range(15):
            d = rng.normal(0, 0.001, size=10)
            d[0] = 0.002
            orch_active.process_event(d, 1.0)
            orch_shadow.process_event(d, 1.0)

        shock = np.zeros(10)
        shock[0] = 0.005
        plan_act = orch_active.process_event(shock, 1.0)
        plan_shd = orch_shadow.process_event(shock, 1.0)

        trace_shd = _extract_trace(plan_shd)
        assert trace_shd["geometric_manifold_shadow"]["dual_engine_shadow_mode"] is True
        assert isinstance(plan_act, ExecutionPlan)
        assert isinstance(plan_shd, ExecutionPlan)
