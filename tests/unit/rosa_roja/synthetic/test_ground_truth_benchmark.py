"""RR-4 ground-truth sandbox: validates Rosa Roja against known dynamics.

Scenarios:
  1. Stable pattern  -> engine must reach EXECUTE and its chosen trajectory
     must predict the true next delta.
  2. Pattern switch  -> engine must detect the regime change (regime_alert)
     shortly after the known `switched_at`.
  3. Recovery        -> after an operator-driven reset (the engine does not
     auto-reset yet; that gap is documented), the engine must re-learn
     pattern B and return to EXECUTE with correct predictions.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.orchestration.rosa_roja.domain.execution import ExecutionPlan
from core.orchestration.rosa_roja.engine import RosaRojaEngine
from core.orchestration.rosa_roja.modules.module1_ingestion import MahalanobisFilter
from core.orchestration.rosa_roja.modules.rhythm_generator import RhythmTrajectoryGenerator
from core.orchestration.rosa_roja.modules.module3_moe_gating import MultiplicativeMoEGating

from .envs import PATTERN_A, PATTERN_B, CyclicPatternEnv, SwitchingPatternEnv


class PassExpert:
    """Always-pass critical expert so gating isolates trajectory dynamics."""

    name = "mock_expert"
    is_critical = True
    threshold = 0.6
    weight = 1.0

    def evaluate_trajectory(self, trajectory) -> float:
        return 0.9

    def update_learning(self, actual, predicted) -> None:
        pass


class StaticDriftSensor:
    name = "static_drift"

    def get_drift_score(self) -> float:
        return 0.0

    def update(self, actual, predicted) -> None:
        pass

    def reset(self) -> None:
        pass


def build_engine() -> RosaRojaEngine:
    return RosaRojaEngine(
        ingestion_filter=MahalanobisFilter(
            noise_threshold=3.0, history_window=40, min_samples_for_cov=8
        ),
        rhythm_generator=RhythmTrajectoryGenerator(
            min_trajectory_len=3, max_trajectory_len=5, top_k=2, oversample_factor=3
        ),
        moe_gating=MultiplicativeMoEGating(),
        expert_jury=[PassExpert()],
        drift_sensors=[StaticDriftSensor()],
    )


class EpisodeReport:
    def __init__(self):
        self.executes = 0
        self.regime_alerts = 0
        self.holds = 0
        self.alert_steps_after_switch: list[int] = []
        self.execute_steps: list[int] = []
        self.prediction_hits = 0
        self.prediction_total = 0

    @property
    def prediction_hit_rate(self) -> float:
        return (
            self.prediction_hits / self.prediction_total
            if self.prediction_total
            else 0.0
        )


def run_episode(engine, env, steps: int, warmup: int = 20) -> EpisodeReport:
    report = EpisodeReport()
    for t in range(steps):
        delta = env.step()
        plan = engine.process_event(delta, 1.0)

        assert isinstance(plan, ExecutionPlan)

        if plan.action == "EXECUTE":
            report.executes += 1
            report.execute_steps.append(t)
            traj = plan.chosen_trajectory
            if len(traj.movements) >= 2 and t >= warmup:
                # Prediction target: the true delta AFTER the one just fed.
                truth = tuple(np.round(env.peek(), 6))
                predicted = tuple(np.round(traj.movements[1].delta_state, 6))
                report.prediction_total += 1
                if predicted == truth:
                    report.prediction_hits += 1
        elif plan.regime_alert:
            report.regime_alerts += 1
            if env.switched_at is not None:
                report.alert_steps_after_switch.append(t - env.switched_at)
        else:
            report.holds += 1
    return report


class TestStablePatternGroundTruth:
    def test_engine_reaches_execute_and_predicts_true_dynamics(self):
        env = CyclicPatternEnv(PATTERN_A)
        engine = build_engine()

        report = run_episode(engine, env, steps=80, warmup=20)

        assert report.executes > 10, f"too few EXECUTE plans: {report.executes}"
        assert report.prediction_total > 5
        assert report.prediction_hit_rate > 0.5, (
            f"trajectory predictions diverge from ground truth: "
            f"{report.prediction_hit_rate:.2f} hit rate"
        )

    def test_no_false_regime_alerts_on_stable_pattern(self):
        env = CyclicPatternEnv(PATTERN_A)
        engine = build_engine()

        report = run_episode(engine, env, steps=80)

        assert report.regime_alerts == 0, "false positive regime alerts on stable pattern"


class TestPatternSwitchDetection:
    def test_switch_is_detected_via_regime_alerts(self):
        env = SwitchingPatternEnv(PATTERN_A, PATTERN_B, switch_step=60)
        engine = build_engine()

        report = run_episode(engine, env, steps=80, warmup=20)

        assert env.switched_at == 60
        assert report.regime_alerts > 0, "regime change went undetected"
        early_alerts = [d for d in report.alert_steps_after_switch if 0 <= d <= 10]
        assert len(early_alerts) >= 2, (
            f"switch detection too slow: alerts at {report.alert_steps_after_switch}"
        )


class TestRecoveryAfterReset:
    def test_engine_relearns_pattern_b_after_operator_reset(self):
        env = SwitchingPatternEnv(PATTERN_A, PATTERN_B, switch_step=60)
        engine = build_engine()

        # Phase 1: learn A, absorb the switch shock (alerts expected).
        run_episode(engine, env, steps=70)

        # Operator-driven reset: the engine cannot auto-reset yet.
        engine.reset()

        # Phase 2: verify re-learning on pure pattern B.
        recovery_env = CyclicPatternEnv(PATTERN_B)
        recovery_report = run_episode(engine, recovery_env, steps=60, warmup=25)

        assert recovery_report.executes > 5, (
            f"no EXECUTE after reset: {recovery_report.executes}"
        )
        assert recovery_report.prediction_total > 3
        assert recovery_report.prediction_hit_rate > 0.4, (
            f"post-reset predictions wrong: {recovery_report.prediction_hit_rate:.2f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
