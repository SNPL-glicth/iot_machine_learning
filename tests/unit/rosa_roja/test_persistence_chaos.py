"""Chaos & peripheral-component persistence tests (F4+F5 of Part 4 §4.6).

Covers acceptance criteria not handled by test_state_persistence.py:
- peripheral engines: Kalman, Taylor tracker, drift detectors
- jury/sensors included in the atomic engine snapshot
- kill -9 simulation with bounded loss
- double concurrent restore (multi-replica readiness)
- Redis transport via fakeredis incl. corrupt payloads at transport level
- checkpoint overhead on the hot path (< 15% on min statistic)
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from core.orchestration.rosa_roja.engine import RosaRojaEngine
from core.orchestration.rosa_roja.modules.module1_ingestion import MahalanobisFilter
from core.orchestration.rosa_roja.modules.rhythm_generator import RhythmTrajectoryGenerator
from core.orchestration.rosa_roja.modules.module3_moe_gating import MultiplicativeMoEGating
from infrastructure.ml.adapters.ml_state_store import InMemoryMLStateStore, RedisMLStateStore
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


class PersistableExpert(PassExpert):
    """Expert whose learning state participates in snapshots."""

    def __init__(self, name="persistable_expert"):
        self.name = name
        self._errors: list[float] = []

    def update_learning(self, actual, predicted) -> None:
        self._errors.append(abs(actual - predicted))

    def export_state(self) -> dict:
        return {"schema_version": 1, "errors": list(self._errors)}

    def import_state(self, payload) -> None:
        if payload.get("schema_version") != 1:
            raise ValueError("bad expert schema")
        self._errors = [float(e) for e in payload["errors"]]


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
        expert_jury=[PersistableExpert()],
        drift_sensors=[StaticDriftSensor()],
    )
    defaults.update(kwargs)
    return RosaRojaEngine(**defaults)


def train_until_execute(engine, env, max_steps=60) -> int:
    for t in range(max_steps):
        plan = engine.process_event(env.step(), 1.0)
        if plan.action == "EXECUTE":
            return t
    raise AssertionError("engine never reached EXECUTE")


class TestKalmanPersistence:
    def test_round_trip_preserves_recent_mae(self):
        from iot_machine_learning.infrastructure.ml.engines.kalman.engine import (
            KalmanPredictionEngine,
        )

        engine = KalmanPredictionEngine(warmup_size=3)
        for p, a in [(1.0, 1.5), (2.0, 2.1), (3.0, 2.6), (4.0, 4.4)]:
            engine.record_actual(p, a)

        restored = KalmanPredictionEngine(warmup_size=3)
        restored.import_state(engine.export_state())

        assert restored.recent_mae() == pytest.approx(engine.recent_mae())
        assert len(restored._error_history) == len(engine._error_history)


class TestTaylorTrackerPersistence:
    def test_engine_delegates_to_tracker(self):
        from iot_machine_learning.infrastructure.ml.engines.taylor.engine import (
            TaylorPredictionEngine,
        )

        source = TaylorPredictionEngine(enable_tracking=True, enable_cache=False)
        for p, a in [(1.0, 1.2), (2.0, 1.7), (3.0, 3.4)]:
            source.record_actual(p, a)

        target = TaylorPredictionEngine(enable_tracking=True, enable_cache=False)
        target.import_state(source.export_state())

        m_source = source._tracker.get_metrics()
        m_target = target._tracker.get_metrics()
        assert m_target.n_samples == m_source.n_samples
        assert m_target.mae == pytest.approx(m_source.mae)


class TestDriftDetectorPersistence:
    def test_page_hinkley_round_trip(self):
        from iot_machine_learning.infrastructure.ml.cognitive.drift.page_hinkley import (
            PageHinkleyConfig, PageHinkleyDetector,
        )

        det = PageHinkleyDetector(PageHinkleyConfig(delta=0.005, lambda_=50.0, alpha=0.999))
        rng = np.random.default_rng(3)
        for _ in range(40):
            det.update(float(rng.normal(10.0, 0.1)))

        restored = PageHinkleyDetector(PageHinkleyConfig(delta=0.005, lambda_=50.0, alpha=0.999))
        restored.import_state(det.export_state())

        assert restored.mean == pytest.approx(det.mean)
        assert restored.cumsum == pytest.approx(det.cumsum)
        assert restored.n_observations == det.n_observations

    def test_adwin_round_trip_and_capacity_guard(self):
        from iot_machine_learning.infrastructure.ml.cognitive.drift.adwin import (
            ADWINDetector,
        )

        det = ADWINDetector(delta=0.002, max_window_size=100)
        rng = np.random.default_rng(5)
        values = [float(rng.normal()) for _ in range(60)]
        for v in values:
            det.update(v)

        restored = ADWINDetector(delta=0.002, max_window_size=100)
        restored.import_state(det.export_state())
        assert list(restored._window) == list(det._window)
        assert restored.window_size == det.window_size

        bad = dict(det.export_state())
        bad["window"] = values * 2  # Exceeds capacity.
        with pytest.raises(ValueError):
            restored.import_state(bad)

    def test_sensor_adapter_per_channel_round_trip(self):
        from infrastructure.ml.adapters.drift_adapter import IoTDriftSensorAdapter

        def make():
            return IoTDriftSensorAdapter(
                name="temp", channels=["ch_a", "ch_b"], detector_type="error_drift",
                window_size=20,
            )

        source, restored = make(), make()
        rng = np.random.default_rng(11)
        for _ in range(30):
            actual, predicted = float(rng.normal()), float(rng.normal(0.2, 0.1))
            source.update(actual, predicted)

        restored.import_state(source.export_state())

        assert source.get_channel_scores().keys() == restored.get_channel_scores().keys()
        for ch in ("ch_a", "ch_b"):
            s_stats = source._detectors[ch].get_stats()
            r_stats = restored._detectors[ch].get_stats()
            assert s_stats["rolling_mae"] == r_stats["rolling_mae"]
            assert s_stats["n_updates"] == r_stats["n_updates"]

    def test_detector_type_mismatch_rejected(self):
        from infrastructure.ml.adapters.drift_adapter import IoTDriftSensorAdapter

        ph = IoTDriftSensorAdapter(name="x", channels=["c"], detector_type="page_hinkley")
        ed = IoTDriftSensorAdapter(name="x", channels=["c"], detector_type="error_drift", window_size=20)
        with pytest.raises(ValueError):
            ed.import_state(ph.export_state())


class TestJuryAndSensorsInSnapshot:
    def test_expert_state_travels_in_engine_snapshot(self):
        store = InMemoryMLStateStore()
        env = SwitchingPatternEnv(PATTERN_A, PATTERN_B, switch_step=10**9)

        expert_a = PersistableExpert()
        engine_a = build_engine(expert_jury=[expert_a], state_store=store, engine_id="jury-test")
        train_until_execute(engine_a, env)
        # Generate feedback so the expert accumulates learning state.
        actual = np.array([1.0])
        predicted = np.array([1.3])
        engine_a.update_feedback(actual, predicted)
        engine_a.checkpoint()
        assert len(expert_a._errors) == 1

        expert_b = PersistableExpert()
        engine_b = build_engine(expert_jury=[expert_b], state_store=store, engine_id="jury-test")
        assert engine_b.restore() is True
        assert expert_b._errors == pytest.approx(expert_a._errors)

    def test_unknown_saved_member_is_loud_failure(self):
        store = InMemoryMLStateStore()
        env = SwitchingPatternEnv(PATTERN_A, PATTERN_B, switch_step=10**9)
        engine_a = build_engine(state_store=store, engine_id="member-test")
        train_until_execute(engine_a, env)
        snapshot = engine_a.export_state()
        snapshot["components"]["jury"] = [{"name": "ghost_expert", "state": {"schema_version": 1}}]
        store.save("member-test", snapshot)

        engine_b = build_engine(state_store=store, engine_id="member-test")
        assert engine_b.restore() is False  # Cold start, no crash.


class TestKillRecovery:
    def test_kill_minus_9_bounds_loss_and_warm_starts(self):
        """Simulates SIGKILL: the engine object is dropped without any
        graceful shutdown hook; only periodic checkpoints survive."""
        interval = 5
        store = InMemoryMLStateStore()
        env = SwitchingPatternEnv(PATTERN_A, PATTERN_B, switch_step=10**9)

        victim = build_engine(state_store=store, engine_id="victim", checkpoint_interval=interval)
        train_until_execute(victim, env)
        events_before_crash = 12
        while victim.state_machine.state.total_events_processed < events_before_crash:
            victim.process_event(env.step(), 1.0)
        watermark_at_last_checkpoint = store.load("victim")["event_watermark"]

        del victim  # SIGKILL: no checkpoint(), no shutdown hooks.

        heir = build_engine(state_store=store, engine_id="victim", checkpoint_interval=interval)
        assert heir.restore() is True
        # Loss bounded by the checkpoint interval.
        assert events_before_crash - watermark_at_last_checkpoint <= interval

        # Heir continues the pattern without re-learning.
        plan = heir.process_event(env.step(), 1.0)
        assert plan.action == "EXECUTE"

    def test_double_restore_is_idempotent(self):
        """Two replicas restoring the same snapshot produce identical decisions."""
        store = InMemoryMLStateStore()
        env_train = SwitchingPatternEnv(PATTERN_A, PATTERN_B, switch_step=10**9)
        writer = build_engine(state_store=store, engine_id="shared")
        train_until_execute(writer, env_train)
        writer.checkpoint()

        next_delta = env_train.step()

        replicas = []
        for _ in range(2):
            r = build_engine(state_store=store, engine_id="shared")
            assert r.restore() is True
            replicas.append(r)

        plans = [replica.process_event(next_delta.copy(), 1.0) for replica in replicas]
        assert all(p.action == "EXECUTE" for p in plans)
        assert plans[0].global_confidence == pytest.approx(plans[1].global_confidence)


class TestRedisTransport:
    def test_redis_store_round_trip_with_fakeredis(self):
        fakeredis = pytest.importorskip("fakeredis")
        fake = fakeredis.FakeRedis(decode_responses=True)
        store = RedisMLStateStore(redis_url="redis://unused")
        store._client = fake  # Inject fake client.

        payload = {"schema_version": 1, "event_watermark": 42}
        assert store.save("engine-x", payload) is True
        loaded = store.load("engine-x")
        assert loaded["event_watermark"] == 42

        assert store.delete("engine-x") is True
        assert store.load("engine-x") is None

    def test_corrupt_backend_payload_degrades_to_cold_start(self):
        fakeredis = pytest.importorskip("fakeredis")
        fake = fakeredis.FakeRedis(decode_responses=True)
        store = RedisMLStateStore(redis_url="redis://unused")
        store._client = fake
        fake.set("mlstate:broken", "{not valid json!!")

        assert store.load("broken") is None  # Transport returns None, not an exception.

        engine = build_engine(state_store=store, engine_id="broken")
        assert engine.restore() is False
        env = SwitchingPatternEnv(PATTERN_A, PATTERN_B, switch_step=10**9)
        train_until_execute(engine, env)  # Fully functional cold.


class TestCheckpointOverhead:
    def test_store_overhead_under_15_percent(self):
        """Criterion §4.6.1-7: checkpoints amortize below 15% of hot path.

        Both engines live through IDENTICAL lifecycles (same event count
        before measuring) so neither is caught in a cheaper phase (early
        HOLDs vs steady-state EXECUTE work differ ~20x). Comparison uses
        min, robust to scheduler noise.
        """
        def build_and_warm(engine_id: str, events: int = 800) -> tuple:
            """Fresh engine + fresh env advanced in lockstep."""
            env = SwitchingPatternEnv(PATTERN_A, PATTERN_B, switch_step=10**9)
            store = InMemoryMLStateStore()
            engine = build_engine(
                state_store=store,
                engine_id=engine_id,
                checkpoint_interval=100,
            )
            for _ in range(events):
                engine.process_event(env.step(), 1.0)
            return engine, env, store

        def measure(engine: RosaRojaEngine, env, iterations: int = 300) -> float:
            latencies = []
            for _ in range(iterations):
                delta = env.step()
                start = time.perf_counter()
                engine.process_event(delta, 1.0)
                latencies.append((time.perf_counter() - start) * 1000)
            return min(latencies)

        # Plain reference: same lifecycle minus persistence.
        env_plain = SwitchingPatternEnv(PATTERN_A, PATTERN_B, switch_step=10**9)
        plain = build_engine()
        for _ in range(800):
            plain.process_event(env_plain.step(), 1.0)

        persisting, env_pers, store = build_and_warm("overhead")

        min_plain = measure(plain, env_plain)
        min_persisting = measure(persisting, env_pers)

        ratio = min_persisting / min_plain
        print(f"\nplain_min={min_plain:.3f}ms persisting_min={min_persisting:.3f}ms ratio={ratio:.3f}")
        assert ratio < 1.15, f"Checkpoint overhead {ratio:.2f}x exceeds budget"

    def test_single_checkpoint_cost_is_bounded(self):
        """A full snapshot write must stay well under one hot-path budget."""
        import cProfile  # noqa: F401  (kept out of the timing loop)
        store = InMemoryMLStateStore()
        env = SwitchingPatternEnv(PATTERN_A, PATTERN_B, switch_step=10**9)
        engine = build_engine(state_store=store, engine_id="ckpt-cost", checkpoint_interval=10**6)
        for _ in range(800):  # Saturated histories = worst-case snapshot size.
            engine.process_event(env.step(), 1.0)

        durations = []
        for _ in range(5):
            start = time.perf_counter()
            engine.checkpoint()
            durations.append((time.perf_counter() - start) * 1000)

        worst = max(durations)
        print(f"\ncheckpoint durations ms: {[round(d, 3) for d in durations]}")
        assert worst < 25.0, f"Single checkpoint {worst:.2f}ms too expensive"
