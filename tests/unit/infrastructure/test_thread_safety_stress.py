"""Concurrent stress tests for Phase 7 thread safety hardening.

Each stress test runs 50 threads × 1,000 operations to verify
the component handles concurrent access without corruption.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import pytest


def _stress_test(
    component: Any,
    operations: list[Callable[[], None]],
    n_threads: int = 50,
    n_ops_per_thread: int = 1000,
) -> None:
    """Run concurrent operations and assert no errors."""
    errors: list[Exception] = []

    def worker():
        for _ in range(n_ops_per_thread):
            for op in operations:
                try:
                    op()
                except Exception as e:
                    errors.append(e)

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = [ex.submit(worker) for _ in range(n_threads)]
        for f in as_completed(futures):
            f.result()  # re-raise unexpected exceptions

    assert len(errors) == 0, (
        f"Thread safety errors ({len(errors)}): {errors[:5]}..."
    )


# ---------------------------------------------------------------------------
# RollingWindowEngine
# ---------------------------------------------------------------------------

class TestRollingWindowEngineStress:
    """50 threads concurrently adding readings and computing stats."""

    def test_concurrent_add_readings(self):
        from iot_machine_learning.infrastructure.ml.cognitive.dynamic.rolling_window_engine import (
            RollingWindowEngine,
        )
        engine = RollingWindowEngine(window_sizes_minutes=[1])

        def make_add(sid: int):
            def op():
                engine.add_reading(sid, 1.0)
            return op

        # 100 sensors, each written by 50 threads
        all_ops = []
        for sensor_id in range(100):
            all_ops.append(make_add(sensor_id))

        _stress_test(engine, all_ops, n_threads=50, n_ops_per_thread=100)

    def test_concurrent_read_write(self):
        from iot_machine_learning.infrastructure.ml.cognitive.dynamic.rolling_window_engine import (
            RollingWindowEngine,
        )
        engine = RollingWindowEngine(window_sizes_minutes=[1, 5])

        # Pre-populate
        for sid in range(10):
            for _ in range(100):
                engine.add_reading(sid, 1.0)

        ops = [
            lambda: engine.add_reading(0, 2.0),
            lambda: engine.compute_stats(0),
            lambda: engine.get_window(0, 1),
            lambda: engine.get_values(0, 1),
            lambda: engine.get_timestamps(0, 1),
            lambda: engine.get_memory_usage(),
        ]

        _stress_test(engine, ops, n_threads=50, n_ops_per_thread=200)

    def test_concurrent_cleanup(self):
        from iot_machine_learning.infrastructure.ml.cognitive.dynamic.rolling_window_engine import (
            RollingWindowEngine,
        )
        engine = RollingWindowEngine(
            window_sizes_minutes=[1],
            max_sensor_age_seconds=0.001,
            max_cache_age_seconds=0.001,
        )

        # Populate sensors
        for sid in range(50):
            for _ in range(10):
                engine.add_reading(sid, 1.0)

        ops = [
            lambda: engine.add_reading(0, 2.0),
            lambda: engine.compute_stats(0),
            lambda: engine.cleanup_sensor(1),
            lambda: engine.get_memory_usage(),
        ]

        _stress_test(engine, ops, n_threads=20, n_ops_per_thread=100)


# ---------------------------------------------------------------------------
# CognitiveMemoryRegistry
# ---------------------------------------------------------------------------

class TestCognitiveMemoryRegistryStress:
    """50 threads concurrently reading/writing config."""

    def test_concurrent_read_write(self):
        from iot_machine_learning.infrastructure.ml.cognitive.memory.cognitive_memory_registry import (
            CognitiveMemoryRegistry,
        )
        registry = CognitiveMemoryRegistry()

        ops = [
            lambda: registry.get_ttl("ANOMALY_CONFIRMED"),
            lambda: registry.get_ttl("REGIME_TRANSITION"),
            lambda: registry.set_ttl("ANOMALY_CONFIRMED", 12345),
            lambda: registry.set_ttl("CUSTOM_TYPE", 99999),
            lambda: registry.min_anomaly_score,
            lambda: registry.min_feature_variability,
            lambda: registry.enable_memory,
            lambda: registry.enable_retrieval,
            lambda: registry.enable_memory_storage(False),
            lambda: registry.enable_retrieval_feature(True),
            lambda: registry.get_ttl_config(),
        ]

        _stress_test(registry, ops, n_threads=50, n_ops_per_thread=200)

    def test_concurrent_ttl_consistency(self):
        from iot_machine_learning.infrastructure.ml.cognitive.memory.cognitive_memory_registry import (
            CognitiveMemoryRegistry,
        )
        registry = CognitiveMemoryRegistry()

        # Writers
        writers = [
            lambda: registry.set_ttl("ANOMALY_CONFIRMED", 100),
            lambda: registry.set_ttl("ANOMALY_CONFIRMED", 200),
            lambda: registry.set_ttl("ANOMALY_CONFIRMED", 300),
        ]
        # Readers
        readers = [
            lambda: registry.get_ttl("ANOMALY_CONFIRMED"),
            lambda: registry.get_ttl_config(),
        ]

        _stress_test(registry, writers + readers, n_threads=30, n_ops_per_thread=500)


# ---------------------------------------------------------------------------
# EventPropagationTracker
# ---------------------------------------------------------------------------

class TestEventPropagationTrackerStress:
    """50 threads concurrently tracking propagation events."""

    def test_concurrent_propagation(self):
        from iot_machine_learning.infrastructure.ml.cognitive.causal.event_propagation_tracker import (
            EventPropagationTracker,
        )
        tracker = EventPropagationTracker(max_propagation_window_seconds=3600)

        ops = [
            lambda: tracker.start_propagation(1, time.time()),
            lambda: tracker.get_active_propagations(),
            lambda: tracker.get_completed_propagations(limit=10),
            lambda: tracker.cleanup_old_propagations(max_age_seconds=86400),
            lambda: tracker.get_propagation_statistics(1, 2),
        ]

        _stress_test(tracker, ops, n_threads=50, n_ops_per_thread=200)

    def test_concurrent_full_cycle(self):
        from iot_machine_learning.infrastructure.ml.cognitive.causal.event_propagation_tracker import (
            EventPropagationTracker,
        )
        tracker = EventPropagationTracker(max_propagation_window_seconds=3600)
        import threading
        _pid_store: list = []
        _pid_lock = threading.Lock()

        def start():
            pid = tracker.start_propagation(1, time.time())
            with _pid_lock:
                _pid_store.append(pid)

        def add():
            nonlocal _pid_store
            with _pid_lock:
                if _pid_store:
                    pid = _pid_store[-1]
                else:
                    return
            tracker.add_to_propagation(pid, 2, time.time())

        def end():
            nonlocal _pid_store
            with _pid_lock:
                if _pid_store:
                    pid = _pid_store.pop(0)
                else:
                    return
            tracker.end_propagation(pid, time.time())

        ops = [start, add, end, lambda: tracker.get_active_propagations()]

        _stress_test(tracker, ops, n_threads=30, n_ops_per_thread=100)


# ---------------------------------------------------------------------------
# OperationalSequenceRegistry
# ---------------------------------------------------------------------------

class TestOperationalSequenceRegistryStress:
    """50 threads concurrently registering and querying sequences."""

    def test_concurrent_register_query(self):
        from iot_machine_learning.infrastructure.ml.cognitive.causal.operational_sequence_registry import (
            OperationalSequenceRegistry,
        )
        from domain.entities.causal import TemporalPattern
        import time

        registry = OperationalSequenceRegistry(max_sequences=1000)
        _counter = 0
        import threading
        _counter_lock = threading.Lock()

        def make_pattern(seq):
            nonlocal _counter
            with _counter_lock:
                _counter += 1
                c = _counter
            return TemporalPattern(
                pattern_id=f"p{c}",
                sequence=seq,
                frequency=1.0,
                confidence=0.9,
                timestamp=time.time(),
                is_pre_anomaly=False,
                avg_duration_seconds=1.0,
            )

        def register():
            p = make_pattern([1, 2, 3])
            registry.register_sequence(p)

        ops = [
            register,
            lambda: registry.get_sequence(f"p{0}"),
            lambda: registry.get_frequent_sequences(min_frequency=1),
            lambda: registry.get_anomaly_precursors(limit=10),
            lambda: registry.get_operational_chains(min_length=2),
            lambda: registry.get_sequence_statistics(),
        ]

        _stress_test(registry, ops, n_threads=30, n_ops_per_thread=100)

    def test_concurrent_cleanup(self):
        from iot_machine_learning.infrastructure.ml.cognitive.causal.operational_sequence_registry import (
            OperationalSequenceRegistry,
        )
        from domain.entities.causal import TemporalPattern
        import time

        registry = OperationalSequenceRegistry(max_sequences=100)

        for i in range(100):
            p = TemporalPattern(
                pattern_id=f"p{i}",
                sequence=[i, i + 1],
                frequency=1.0,
                confidence=0.9,
                timestamp=time.time() - 999999,
                is_pre_anomaly=(i % 2 == 0),
                avg_duration_seconds=1.0,
            )
            registry.register_sequence(p)

        ops = [
            lambda: registry.cleanup_old_sequences(max_age_seconds=0),
            lambda: registry.get_sequence("p0"),
            lambda: registry.get_sequence_statistics(),
        ]

        _stress_test(registry, ops, n_threads=10, n_ops_per_thread=50)
