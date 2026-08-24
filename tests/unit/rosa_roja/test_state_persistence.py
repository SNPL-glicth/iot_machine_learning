"""Persistence & warm-start tests for Rosa Roja ML state.

Covers the acceptance criteria from AUDIT_REPORT_PART_4 §4.6.1:
- exact round-trip of component state
- warm start: EXECUTE on first post-restore event, no re-learning window
- Mahalanobis active immediately after restore
- corrupt payload -> cold start without crash
- bounded loss window via periodic checkpoints
"""

from __future__ import annotations

import numpy as np
import pytest

from core.orchestration.rosa_roja.domain.theta_belief import ThetaBelief
from core.orchestration.rosa_roja.domain.state_machine import (
    IngestionState,
    PipelineState,
    StateMachine,
)
from core.orchestration.rosa_roja.engine import RosaRojaEngine
from core.orchestration.rosa_roja.modules.module1_ingestion import MahalanobisFilter
from core.orchestration.rosa_roja.modules.rhythm_generator import RhythmTrajectoryGenerator
from core.orchestration.rosa_roja.modules.module3_moe_gating import MultiplicativeMoEGating
from infrastructure.ml.adapters.ml_state_store import InMemoryMLStateStore
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


def train_until_execute(engine: RosaRojaEngine, env: SwitchingPatternEnv, max_steps=60):
    """Feed the pattern until the engine reaches its first EXECUTE."""
    for t in range(max_steps):
        delta = env.step()
        plan = engine.process_event(delta, 1.0)
        if plan.action == "EXECUTE":
            return t
    raise AssertionError("engine never reached EXECUTE during training")


class TestThetaBeliefPersistence:
    def test_round_trip_preserves_posterior(self):
        theta = ThetaBelief(alpha=0.95)
        theta.update((1.0, 0.0), (0.0, 2.0))
        theta.update((1.0, 0.0), (0.0, 2.0))
        theta.update((0.0, 2.0), (1.0, 0.0))
        expected = theta.get_transition_probabilities((1.0, 0.0))

        restored = ThetaBelief()
        restored.import_state(theta.export_state())

        assert restored.total_updates == theta.total_updates
        assert restored.get_transition_probabilities((1.0, 0.0)) == expected

    def test_unknown_schema_raises(self):
        with pytest.raises(ValueError):
            ThetaBelief().import_state({"schema_version": 99})

    def test_import_does_not_replay_decay(self):
        theta = ThetaBelief(alpha=0.5)
        theta.update((1.0,), (2.0,))
        snapshot = theta.export_state()

        replayed = ThetaBelief(alpha=0.5)
        replayed.import_state(snapshot)
        # A replay through update() would have total_updates == 2 and decayed weights.
        assert replayed.total_updates == 1
        probs = replayed.get_transition_probabilities((1.0,))
        assert abs(sum(probs.values()) - 1.0) < 1e-9
        assert probs[(2.0,)] == pytest.approx(1.0)


class TestMahalanobisWarmStart:
    def test_round_trip_and_immediate_outlier_detection(self):
        rng = np.random.default_rng(7)
        source = MahalanobisFilter(noise_threshold=3.0, min_samples_for_cov=8)
        for _ in range(30):
            source.process_raw_step(rng.normal([1.0, 0.5], 0.05), 1.0)

        payload = source.export_state()
        restored = MahalanobisFilter(noise_threshold=3.0, min_samples_for_cov=8)
        restored.import_state(payload)

        assert restored._n == source._n
        assert np.allclose(restored._mean, source._mean)
        assert np.allclose(restored._cov_inv, source._cov_inv)
        assert len(restored._history) == len(source._history)

        # Warm filter flags an extreme outlier immediately; a cold filter
        # would return d=0.0 until min_samples are re-learned.
        extreme = np.array([50.0, -40.0])
        _, hot_outlier = restored.process_raw_step(extreme, 1.0)
        assert hot_outlier is True

    def test_inconsistent_payload_rejected(self):
        with pytest.raises(ValueError):
            MahalanobisFilter().import_state({"schema_version": 1, "n": 5})


class TestEngineWarmStart:
    def test_first_post_restore_event_executes(self):
        env_train = SwitchingPatternEnv(PATTERN_A, PATTERN_B, switch_step=10**9)
        engine_a = build_engine()
        train_until_execute(engine_a, env_train)
        snapshot = engine_a.export_state()

        engine_b = build_engine()  # Fresh, cold modules.
        engine_b.import_state(snapshot)

        # Learning state carried over.
        assert engine_b.state_machine.state.total_events_processed == (
            engine_a.state_machine.state.total_events_processed
        )
        assert engine_b.auto_reset_count == engine_a.auto_reset_count

        # The very next movement of the known pattern executes immediately:
        # covariance active (no re-learning window) and history dense enough
        # to generate candidates.
        delta = env_train.step()
        plan = engine_b.process_event(delta, 1.0)
        assert plan.action == "EXECUTE"

    def test_restore_via_store_round_trip(self):
        store = InMemoryMLStateStore()
        env = SwitchingPatternEnv(PATTERN_A, PATTERN_B, switch_step=10**9)
        engine_a = build_engine(state_store=store, engine_id="rr-test")
        train_until_execute(engine_a, env)
        engine_a.checkpoint()

        engine_b = build_engine(state_store=store, engine_id="rr-test")
        assert engine_b.restore() is True
        plan = engine_b.process_event(env.step(), 1.0)
        assert plan.action == "EXECUTE"

    def test_corrupt_snapshot_degrades_to_cold_start(self):
        store = InMemoryMLStateStore()
        store.save("rr-corrupt", {"schema_version": 999, "components": {}})

        engine = build_engine(state_store=store, engine_id="rr-corrupt")
        assert engine.restore() is False

        # Engine remains fully functional in cold-start mode.
        env = SwitchingPatternEnv(PATTERN_A, PATTERN_B, switch_step=10**9)
        train_until_execute(engine, env)

    def test_missing_snapshot_returns_false(self):
        engine = build_engine(state_store=InMemoryMLStateStore(), engine_id="absent")
        assert engine.restore() is False

    def test_no_store_checkpoint_is_noop(self):
        engine = build_engine()
        assert engine.checkpoint() is False
        assert engine.restore() is False


class TestCheckpointInterval:
    def test_auto_checkpoints_bound_loss_window(self):
        store = InMemoryMLStateStore()
        env = SwitchingPatternEnv(PATTERN_A, PATTERN_B, switch_step=10**9)
        engine = build_engine(
            state_store=store,
            engine_id="rr-interval",
            checkpoint_interval=5,
        )

        for _ in range(11):
            engine.process_event(env.step(), 1.0)

        saved = store.load("rr-interval")
        assert saved is not None
        watermark = saved["event_watermark"]
        # Loss on kill -9 bounded by the interval.
        assert 0 < watermark <= 11
        assert 11 - watermark < 5

    def test_state_machine_audit_trail_survives(self):
        sm = StateMachine(outlier_reset_threshold=3)
        sm.transition(new_pipeline_state=PipelineState.GENERATING, reason="boot")
        sm.on_outlier_detected()

        restored = StateMachine(outlier_reset_threshold=3)
        restored.import_state(sm.export_state())

        assert restored.pipeline_state == PipelineState.GENERATING
        assert restored.state.consecutive_outliers == 1
        assert len(restored.state.transition_history) >= 1
        assert restored.state.transition_history[0].reason == "boot"
