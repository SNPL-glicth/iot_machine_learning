"""Invariance test suite for Geometric Manifold Shadow Mode.

Validates that over >= 30 simulated market cycles:
1. Orchestrator with GeometricManifoldAdapter in manifold_shadow_mode=True produces
   100% identical actions (EXECUTE/HOLD) and global confidence as running with
   manifold_engine=None.
2. The decision_trace contains complete diagnostic geometric_manifold_shadow telemetry.
3. When manifold_shadow_mode=False, positive divergence strictly suppresses confidence
   and can veto marginal execution plans end-to-end.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock
import numpy as np
import pytest

from domain.entities.rosa_roja.execution import ActionEnvelope, ExecutionPlan
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
from iot_machine_learning.infrastructure.ml.master_engine.geometric_manifold_adapter import (
    GeometricManifoldAdapter,
)
from iot_machine_learning.infrastructure.ml.master_engine.plan_builder import (
    build_orchestrated_execution_plan,
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


def _extract_trace(plan: ExecutionPlan) -> dict:
    if plan.envelope and plan.envelope.metadata:
        return plan.envelope.metadata.get("decision_trace", {})
    if getattr(plan, "veto_details", None) and isinstance(plan.veto_details, dict):
        return plan.veto_details.get("decision_trace", {})
    return {}


class TestManifoldShadowInvariance:
    """Verifies invariant behavior of manifold_shadow_mode=True vs manifold_engine=None."""

    def test_thirty_five_cycles_manifold_shadow_invariance(self) -> None:
        """Runs 35 cycles: manifold_shadow_mode=True must be 100% identical to manifold=None."""
        orch_none = MasterEquationOrchestrator(
            rosa_roja_engine=_build_rosa_roja_engine(),
            risk_adapter=RiskEngineAdapter(l_max=0.03, default_sigma=0.002),
            temporal_adapter=TemporalEngineAdapter(),
            manifold_engine=None,
            shadow_mode=False,
        )
        orch_shadow = MasterEquationOrchestrator(
            rosa_roja_engine=_build_rosa_roja_engine(),
            risk_adapter=RiskEngineAdapter(l_max=0.03, default_sigma=0.002),
            temporal_adapter=TemporalEngineAdapter(),
            manifold_engine=GeometricManifoldAdapter(),
            manifold_shadow_mode=True,
            shadow_mode=False,
        )

        rng = np.random.default_rng(12345)
        for step in range(35):
            delta = rng.normal(0, 0.0015, size=10)
            delta[0] = 0.004
            delta[5] = float(rng.uniform(-0.5, 0.5))
            dt = 1.0

            plan_none = orch_none.process_event(delta, dt)
            plan_shadow = orch_shadow.process_event(delta, dt)

            # Invariance Check 1: Action (EXECUTE / HOLD) must be 100% identical
            assert plan_shadow.action == plan_none.action, f"Cycle {step}: Action mismatch"

            # Invariance Check 2: Global confidence must be identical to machine precision
            assert plan_shadow.global_confidence == pytest.approx(
                plan_none.global_confidence, abs=1e-12
            ), f"Cycle {step}: Confidence mismatch"

            # Invariance Check 3: If executed, target magnitude must not be overridden
            if plan_shadow.action == "EXECUTE" and plan_shadow.envelope and plan_none.envelope:
                assert plan_shadow.envelope.magnitude == pytest.approx(
                    plan_none.envelope.magnitude, abs=1e-12
                )

            # Invariance Check 4: Diagnostic shadow telemetry is present in shadow mode
            trace_none = _extract_trace(plan_none)
            trace_shadow = _extract_trace(plan_shadow)
            assert "geometric_manifold_shadow" not in trace_none or trace_none["geometric_manifold_shadow"] is None
            geo_shadow = trace_shadow.get("geometric_manifold_shadow")
            assert geo_shadow is not None, f"Cycle {step}: Missing geometric_manifold_shadow trace"
            assert geo_shadow["manifold_shadow_mode"] is True
            assert geo_shadow["nominal_certeza"] == pytest.approx(trace_none["certeza"], abs=1e-9)

    def test_active_mode_positive_divergence_suppression_and_veto_end_to_end(self) -> None:
        """When manifold_shadow_mode=False, div > 0 suppresses certainty and can veto execution."""
        # 1. State with positive Liouville divergence Tr(J) > 0
        comp_shadow = compute_master_equation(
            phi_moe_base=0.20, i_cvar=1.0, lambda_t_crono=1.0, kuramoto_r=0.05,
            phase_alignment=1.0, manifold_engine=GeometricManifoldAdapter(), mahalanobis_d=0.1,
            manifold_shadow_mode=True,
        )
        comp_active = compute_master_equation(
            phi_moe_base=0.20, i_cvar=1.0, lambda_t_crono=1.0, kuramoto_r=0.05,
            phase_alignment=1.0, manifold_engine=GeometricManifoldAdapter(), mahalanobis_d=0.1,
            manifold_shadow_mode=False,
        )

        assert comp_shadow.divergence > 0.0
        assert comp_shadow.certeza == pytest.approx(0.20 * 0.05, abs=1e-9)
        assert comp_active.certeza < comp_shadow.certeza
        expected_active = comp_shadow.certeza * math.exp(-comp_active.divergence)
        assert comp_active.certeza == pytest.approx(expected_active, abs=1e-6)

        # 2. End-to-end plan validation: Liouville suppression turns EXECUTE into HOLD
        gamma_exec = 0.009
        envelope = ActionEnvelope(magnitude=1.0, bounds={}, max_steps=10, metadata={})
        plan_candidate = ExecutionPlan(
            action="EXECUTE", chosen_trajectory=MagicMock(), global_confidence=0.8,
            envelope=envelope, invalidation_step=None, regime_alert=False, veto_details={},
        )

        plan_shadow = build_orchestrated_execution_plan(
            plan_base=plan_candidate, certeza=comp_shadow.certeza, magnitud_objetivo=0.01,
            momentum_veto=1.0, master_trace={"decision_trace": comp_shadow.geometric_manifold_shadow},
            shadow_mode=False, gamma_exec=gamma_exec,
        )
        plan_active = build_orchestrated_execution_plan(
            plan_base=plan_candidate, certeza=comp_active.certeza, magnitud_objetivo=0.01,
            momentum_veto=1.0, master_trace={"decision_trace": comp_active.geometric_manifold_shadow},
            shadow_mode=False, gamma_exec=gamma_exec,
        )

        assert plan_shadow.action == "EXECUTE"
        assert plan_shadow.global_confidence == pytest.approx(comp_shadow.certeza, abs=1e-9)
        assert plan_active.action == "HOLD"
        assert plan_active.veto_details["reason"] == "Certeza_Below_Gamma_Exec"
