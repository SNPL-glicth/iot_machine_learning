"""RR-3 tests: reactive trajectory invalidation & automatic regime recovery."""

from __future__ import annotations

import numpy as np
import pytest

from core.orchestration.rosa_roja.domain.movement import Movement, RhythmSignature
from core.orchestration.rosa_roja.domain.execution import ExecutionPlan
from core.orchestration.rosa_roja.engine import RosaRojaEngine
from core.orchestration.rosa_roja.modules.module1_ingestion import MahalanobisFilter
from core.orchestration.rosa_roja.modules.rhythm_generator import RhythmTrajectoryGenerator
from core.orchestration.rosa_roja.modules.module3_moe_gating import MultiplicativeMoEGating
from tests.unit.rosa_roja.synthetic.envs import PATTERN_A, PATTERN_B, SwitchingPatternEnv


class PassExpert:
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


def make_movement(delta, velocity: float = 1.0) -> Movement:
    d = np.asarray(delta, dtype=float)
    norm = np.linalg.norm(d)
    direction = d / norm if norm > 0 else np.zeros_like(d)
    rhythm = RhythmSignature(
        tempo_ratio=1.0,
        velocity_delta=0.0,
        acceleration=0.0,
        phase_angle=0.0,
        entropy_rate=0.1,
    )
    return Movement(
        delta_state=d,
        delta_time=1.0,
        velocity=velocity,
        direction=direction,
        rhythm_signature=rhythm,
        mahalanobis_distance=1.0,
        timestamp=0.0,
    )


def build_engine(**kwargs) -> RosaRojaEngine:
    defaults = dict(
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
    defaults.update(kwargs)
    return RosaRojaEngine(**defaults)


class TestReactiveTrajectoryInvalidation:
    """Mid-trajectory deviation must immediately invalidate the execution plan."""

    def test_deviation_triggers_emergency_flush(self):
        engine = build_engine()
        env = SwitchingPatternEnv(PATTERN_A, PATTERN_B, switch_step=1000)  # Never switches

        # Phase 1: learn stable pattern until first EXECUTE
        first_execute_step = None
        for t in range(60):
            delta = env.step()
            plan = engine.process_event(delta, 1.0)
            if plan.action == "EXECUTE":
                first_execute_step = t
                break

        assert first_execute_step is not None, "never reached EXECUTE"

        # Next step should be valid (matches predicted trajectory)
        correct_delta = env.step()
        plan = engine.process_event(correct_delta, 1.0)
        assert plan.action == "EXECUTE"

        # Now inject a DEVIATING movement: same magnitude, orthogonal axis
        # Pattern A alternates x/y axes. Find true next, then rotate axis.
        true_next = env.peek()
        # Rotate by swapping non-zero components
        rotated = np.roll(true_next, 1)  # e.g., x-axis -> y-axis, same magnitude
        deviating_delta = rotated.astype(float)

        plan = engine.process_event(deviating_delta, 1.0)

        assert plan.action in {"EMERGENCY_FLUSH", "HOLD"}
        assert "Deviation" in plan.veto_details.get("reason", "")

    def test_velocity_deviation_triggers_hold(self):
        engine = build_engine()
        env = SwitchingPatternEnv(PATTERN_A, PATTERN_B, switch_step=1000)

        # Reach EXECUTE
        for _ in range(60):
            delta = env.step()
            if engine.process_event(delta, 1.0).action == "EXECUTE":
                break

        # Valid step
        delta = env.step()
        assert engine.process_event(delta, 1.0).action == "EXECUTE"

        # Velocity deviation: same direction, 10x magnitude
        true_next = env.peek()
        velocity_delta = true_next * 10.0
        plan = engine.process_event(velocity_delta, 1.0)

        assert plan.action in {"EMERGENCY_FLUSH", "HOLD"}
        assert "Velocity" in plan.veto_details.get("reason", "")

    def test_valid_step_continues_execution(self):
        engine = build_engine()
        env = SwitchingPatternEnv(PATTERN_A, PATTERN_B, switch_step=1000)

        # Reach EXECUTE
        for _ in range(60):
            delta = env.step()
            if engine.process_event(delta, 1.0).action == "EXECUTE":
                break

        # Follow the true pattern for several steps - all should be EXECUTE
        for _ in range(5):
            delta = env.step()
            plan = engine.process_event(delta, 1.0)
            assert plan.action == "EXECUTE"


class TestAutoRegimeRecovery:
    """After 3 consecutive outliers, engine must auto-reset and recover on new regime."""

    def test_auto_resets_after_3_outliers_and_recovers_on_pattern_b(self):
        # Use stricter noise threshold to ensure switch causes outliers
        engine = build_engine(
            ingestion_filter=MahalanobisFilter(
                noise_threshold=3.0, history_window=40, min_samples_for_cov=8
            ),
        )
        env = SwitchingPatternEnv(PATTERN_A, PATTERN_B, switch_step=60)

        auto_reset_seen = False
        executes_after_switch = 0
        hits_after_switch = 0
        total_preds_after_switch = 0

        for t in range(140):
            delta = env.step()
            plan = engine.process_event(delta, 1.0)

            if plan.action == "HOLD" and plan.veto_details.get("reason") == "Auto_Regime_Reset_Triggered":
                auto_reset_seen = True

            if t > 60 and plan.action == "EXECUTE":
                executes_after_switch += 1
                traj = plan.chosen_trajectory
                if len(traj.movements) >= 2:
                    total_preds_after_switch += 1
                    predicted = tuple(np.round(traj.movements[1].delta_state, 6))
                    truth = tuple(np.round(env.peek(), 6))
                    if predicted == truth:
                        hits_after_switch += 1

        assert auto_reset_seen, "auto regime reset never triggered"
        assert engine.auto_reset_count == 1, f"expected 1 auto reset, got {engine.auto_reset_count}"
        assert executes_after_switch > 5, f"too few executes after switch: {executes_after_switch}"
        assert total_preds_after_switch > 3
        hit_rate = hits_after_switch / total_preds_after_switch
        assert hit_rate > 0.4, f"post-recovery predictions wrong: {hit_rate:.2f}"

    def test_no_perpetual_alerts_after_auto_reset(self):
        """After auto-reset, regime_alert must stop (not perpetual)."""
        engine = build_engine(
            ingestion_filter=MahalanobisFilter(
                noise_threshold=3.0, history_window=40, min_samples_for_cov=8
            ),
        )
        env = SwitchingPatternEnv(PATTERN_A, PATTERN_B, switch_step=60)

        alerts_after_reset = 0
        auto_reset_step = None

        for t in range(140):
            delta = env.step()
            plan = engine.process_event(delta, 1.0)

            if plan.veto_details.get("reason") == "Auto_Regime_Reset_Triggered":
                auto_reset_step = t

            if auto_reset_step is not None and t > auto_reset_step and plan.regime_alert:
                alerts_after_reset += 1

        # After full auto-reset, a brief adaptation window (min_samples_for_cov=8)
        # may have a few alerts while covariance rebuilds, but then stops.
        assert alerts_after_reset <= 10, f"too many alerts after auto-reset: {alerts_after_reset}"

    def test_explicit_reset_still_works(self):
        """Manual engine.reset() must clear everything including new counters."""
        engine = build_engine()
        env = SwitchingPatternEnv(PATTERN_A, PATTERN_B, switch_step=60)

        for _ in range(80):
            delta = env.step()
            engine.process_event(delta, 1.0)

        assert engine.auto_reset_count >= 1

        engine.reset()

        assert engine.auto_reset_count == 0
        assert engine._consecutive_outliers == 0
        assert not engine._tracker.has_active_trajectory


class TestConsecutiveOutlierCounting:
    """Counter must increment on outliers/deviations and reset on valid steps."""

    def test_counter_resets_on_valid_step(self):
        """Non-outlier step must reset the consecutive outlier counter."""
        engine = build_engine(
            ingestion_filter=MahalanobisFilter(
                noise_threshold=5.0, history_window=20, min_samples_for_cov=5
            ),
        )
        # Feed normal pattern to build history
        for i in range(10):
            engine.process_event(np.array([float(i), 0.0, 0.0]), 1.0)

        # Now a large outlier
        engine.process_event(np.array([100.0, 100.0, 100.0]), 1.0)
        assert engine._consecutive_outliers == 1

        # Normal step well within distribution resets it
        engine.process_event(np.array([5.0, 0.0, 0.0]), 1.0)
        assert engine._consecutive_outliers == 0

    def test_deviation_increments_counter(self):
        """Tracker invalidation must increment the outlier counter."""
        engine = build_engine()
        env = SwitchingPatternEnv(PATTERN_A, PATTERN_B, switch_step=1000)

        for _ in range(40):
            delta = env.step()
            if engine.process_event(delta, 1.0).action == "EXECUTE":
                break

        # Valid step
        engine.process_event(env.step(), 1.0)
        assert engine._consecutive_outliers == 0

        # Deviation - first invalidation increments counter
        true_next = env.peek()
        rotated = np.roll(true_next, 1)
        plan = engine.process_event(rotated.astype(float), 1.0)
        assert engine._consecutive_outliers == 1
        assert plan.action in {"EMERGENCY_FLUSH", "HOLD"}

        # After deviation, tracker is cleared.
        # Next deviation has no active trajectory to compare against,
        # so it goes through normal pipeline (not a Mahalanobis outlier).
        # Counter resets on the non-outlier pipeline step.
        plan2 = engine.process_event(rotated.astype(float), 1.0)
        assert engine._consecutive_outliers == 0
        # The second deviating step may now be treated as a fresh pattern
        # and could even EXECUTE if it happens to match some branch.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])