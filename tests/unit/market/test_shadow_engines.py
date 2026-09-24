"""Unit tests for Stochastic Risk Engine and Fractal Chronometric Engine in pure shadow mode.

Validates:
1. RiskEngineAdapter formulas (R_t, Omega_t, Omega_max clip, CVaR, veto_riesgo).
2. TemporalEngineAdapter Option A (zero-crossings, dR/dt, dS/dt, Lambda(t)).
3. Non-interference invariant: shadow experts do not alter phi_moe or execution decisions.
"""

from typing import Any
import math
import numpy as np
import pytest

from iot_machine_learning.infrastructure.ml.adapters import (
    RiskEngineAdapter,
    TemporalEngineAdapter,
    TaylorExpertAdapter,
    KalmanExpertAdapter,
    StatisticalExpertAdapter,
)
from iot_machine_learning.infrastructure.ml.engines.kalman.engine import KalmanPredictionEngine
from iot_machine_learning.domain.entities.rosa_roja.movement import Movement
from iot_machine_learning.domain.entities.rosa_roja.trajectory import Trajectory, TerminalState
from iot_machine_learning.infrastructure.ml.engines.rosa_roja.algorithms.engine import RosaRojaEngine
from iot_machine_learning.infrastructure.ml.engines.rosa_roja.algorithms.modules.module1_ingestion import MahalanobisFilter
from iot_machine_learning.infrastructure.ml.engines.rosa_roja.algorithms.modules.module3_moe_gating import MultiplicativeMoEGating
from iot_machine_learning.infrastructure.ml.engines.rosa_roja.algorithms.modules.rhythm_generator import RhythmTrajectoryGenerator
from iot_machine_learning.infrastructure.ml.engines.statistical import StatisticalPredictionEngine
from iot_machine_learning.infrastructure.ml.engines.taylor.engine import TaylorPredictionEngine
from iot_machine_learning.infrastructure.ml.master_engine import MasterEquationOrchestrator


class TestRiskEngineAdapter:
    """Validates the Stochastic Risk Engine mathematics and bounds."""

    def test_tolerance_tube_and_omega_clip(self) -> None:
        adapter = RiskEngineAdapter(omega_alpha=0.2, omega_max=2.0, l_max=0.05, default_sigma=0.01)

        # Baseline: 0 breaches in return space
        v0 = adapter.record_observation(return_signal=0.001, delta_time=1.0, log_return=0.001, expected_return=0.001)
        assert v0["omega_t"] == 0.0
        assert v0["R_t"] == pytest.approx(0.01 * 1.0 * math.exp(0.0), rel=1e-3)
        assert v0["veto_riesgo"] == 1

        # Simulate 20 massive breaches far exceeding R_t in return space
        for i in range(20):
            # Return signal jumps by 0.05 (500 bps), far above R_t ~ 0.01-0.07
            v = adapter.record_observation(return_signal=0.001 + (i + 1) * 0.05, delta_time=1.0, log_return=0.01, expected_return=0.001)

        # Binary breach EMA converges to 1.0
        assert v["omega_t"] == pytest.approx(1.0, abs=0.02)
        assert v["R_t"] == pytest.approx(v["sigma_t"] * 1.0 * math.exp(v["omega_t"]), rel=1e-3)

        # If a regime shock sets omega above omega_max (e.g. 2.5), it must clip at omega_max = 2.0
        adapter._omega = 2.5
        v_clip = adapter.record_observation(return_signal=0.05, delta_time=1.0, log_return=0.01, expected_return=0.05)
        assert v_clip["omega_t"] == pytest.approx(2.0, rel=1e-3)
        assert adapter.cvar_multiplier == 2.70
        assert v_clip["cvar_t"] == pytest.approx(2.70 * v_clip["R_t"], rel=1e-3)

        # Backwards compatibility alias check
        v_alias = adapter.record_observation(current_price=0.01, delta_time=1.0, log_return=0.01, expected_price=0.01)
        assert "R_t" in v_alias and v_alias["R_t"] > 0

    def test_cvar_capitulation_veto(self) -> None:
        """When CVaR exceeds l_max, veto_riesgo must transition to 0 (veto)."""
        adapter = RiskEngineAdapter(l_max=0.02, default_sigma=0.02)

        # High volatility / breach -> cvar ~ 2.0627 * 0.02 * exp(1.5) ~ 0.185 > 0.02
        adapter._omega = 1.5
        v = adapter.record_observation(return_signal=0.001, delta_time=1.0, log_return=0.03, expected_return=0.001)
        assert v["cvar_t"] > 0.02
        assert v["veto_riesgo"] == 0

    def test_expert_jury_port_contract(self) -> None:
        adapter = RiskEngineAdapter()
        assert adapter.name == "stochastic_risk"
        assert adapter.weight == 0.0  # Zero weight in shadow mode
        assert adapter.is_critical is False

        deltas = np.array([[0.001 * i] + [0] * 9 for i in range(12)])
        movements = [Movement.from_raw(deltas[i], delta_time=1.0, timestamp=float(i)) for i in range(12)]
        traj = Trajectory(
            movements=tuple(movements),
            coherence_score=0.03,
            invalidation_step=None,
            terminal_state=TerminalState(state_vector=deltas[-1], step_index=11, confidence=0.03),
        )
        conf = adapter.evaluate_trajectory(traj)
        assert 0.20 <= conf <= 0.95

    def test_scale_guardrail_warns_on_price_magnitude(self, caplog: pytest.LogCaptureFixture) -> None:
        """When an input with magnitude > 1.0 (e.g. dollar price 512.45) is passed, a warning is logged."""
        import logging
        adapter = RiskEngineAdapter()
        with caplog.at_level(logging.WARNING):
            adapter.record_observation(current_price=512.45)
        assert any("magnitude" in r.message for r in caplog.records)


class TestTemporalEngineAdapter:
    """Validates the Fractal Chronometric Engine mathematics and synchrony index."""

    def test_microstructure_periodicity_zero_crossing(self) -> None:
        adapter = TemporalEngineAdapter(window_size=20)

        # Alternating book imbalance: +0.5, -0.5, +0.5, -0.5 (period = 2.0)
        for i in range(20):
            imb = 0.5 if i % 2 == 0 else -0.5
            v = adapter.record_observation(current_price=100.0 + 0.01 * i, book_imbalance=imb)

        assert 2.0 <= v["dominant_period"] <= 4.0
        assert v["dR_dt"] > 0.0

    def test_synchrony_resonance_index(self) -> None:
        """When dS/dt equals dR/dt, Lambda(t) must approach 1.0 (resonance)."""
        adapter = TemporalEngineAdapter()

        # Seed imbalance buffer
        for i in range(15):
            adapter.record_observation(current_price=100.0, book_imbalance=0.2 * math.sin(i * 0.5))

        metrics = adapter.get_shadow_metrics()
        dR_dt = metrics["dR_dt"]

        # Exact match: price velocity == dR_dt
        v_res = adapter.record_observation(current_price=100.0, price_velocity=dR_dt)
        assert v_res["lambda_crono"] >= 0.95  # Perfect synchrony

        # Severe discrepancy: price velocity is 10x dR_dt
        v_desync = adapter.record_observation(current_price=100.0, price_velocity=10.0 * dR_dt)
        assert v_desync["lambda_crono"] <= 0.05  # Severe desynchronization


class TestShadowModeIsolation:
    """Verifies that shadow engines do not interfere with live execution or phi_moe."""

    def _build_engine(self, with_shadow: bool = True) -> Any:
        ingestion = MahalanobisFilter(noise_threshold=3.0, history_window=50, min_samples_for_cov=10)
        rhythm = RhythmTrajectoryGenerator(min_trajectory_len=11, max_trajectory_len=15, top_k=4)
        gating = MultiplicativeMoEGating(variance_penalty=0.5)
        jury = [
            TaylorExpertAdapter(engine=TaylorPredictionEngine()),
            KalmanExpertAdapter(engine=KalmanPredictionEngine()),
            StatisticalExpertAdapter(engine=StatisticalPredictionEngine()),
        ]
        engine = RosaRojaEngine(
            ingestion_filter=ingestion,
            rhythm_generator=rhythm,
            moe_gating=gating,
            expert_jury=jury,
            drift_sensors=[],
        )
        if not with_shadow:
            return engine
        return MasterEquationOrchestrator(
            rosa_roja_engine=engine,
            risk_adapter=RiskEngineAdapter(),
            temporal_adapter=TemporalEngineAdapter(),
            shadow_mode=True,
        )

    def test_non_interference_invariant(self) -> None:
        """Execution plans and phi_moe must be 100% identical with or without shadow experts."""
        engine_clean = self._build_engine(with_shadow=False)
        engine_shadow = self._build_engine(with_shadow=True)

        rng = np.random.default_rng(123)
        for step in range(25):
            delta = rng.normal(0, 0.002, size=10)
            # Add linear trend to feature 0 so trajectory generation succeeds
            delta[0] = 0.005
            dt = 1.0

            plan_clean = engine_clean.process_event(delta, dt)
            plan_shadow = engine_shadow.process_event(delta, dt)

            # 1. Actions must be identical
            assert plan_clean.action == plan_shadow.action, f"Step {step}: Action mismatch {plan_clean.action} vs {plan_shadow.action}"

            # 2. Global confidence (phi_moe) must be mathematically identical
            assert plan_clean.global_confidence == pytest.approx(
                plan_shadow.global_confidence, abs=1e-12
            ), f"Step {step}: Confidence divergence"

            # 3. Decision trace in shadow engine must contain shadow payloads
            trace_shadow = plan_shadow.envelope.metadata.get("decision_trace") if plan_shadow.envelope else (
                plan_shadow.veto_details.get("decision_trace") if isinstance(plan_shadow.veto_details, dict) else None
            )
            if trace_shadow:
                assert "risk_engine_shadow" in trace_shadow
                assert "temporal_engine_shadow" in trace_shadow
