"""High-concurrency stress tests — 100 → 500 → 1000 concurrent series.

Validates:
- latency (p99 < 500ms per operation)
- error rate (< 1%)
- stability (no crashes, no deadlocks)

Uses only stdlib + project code. No external load-test frameworks.
"""

from __future__ import annotations

import concurrent.futures
import logging
import random
import threading
import time
import tracemalloc
from collections import deque
from typing import Dict, List, Tuple

import pytest

from iot_machine_learning.domain.entities.iot.sensor_reading import Reading
from iot_machine_learning.infrastructure.ml.cognitive.bayesian_weight_tracker import (
    BayesianWeightTracker,
)
from iot_machine_learning.infrastructure.ml.cognitive.fusion.engine_selector import (
    WeightedFusion,
)
from iot_machine_learning.infrastructure.ml.anomaly.core.detector import (
    VotingAnomalyDetector,
)
from iot_machine_learning.infrastructure.ml.anomaly.core.config import (
    AnomalyDetectorConfig,
)
from iot_machine_learning.ml_service.consumers.sliding_window import (
    SlidingWindowStore,
)

logger = logging.getLogger(__name__)

# Reproducible seeds
_RANDOM_SEED = 42


def _generate_series_id(i: int) -> str:
    return f"stress_series_{i}"


def _make_reading(series_id: str, t: float, value: float) -> Reading:
    return Reading(
        series_id=series_id,
        value=value,
        timestamp=t,
    )


def _synthetic_values(n: int, noise: float = 0.5) -> List[float]:
    random.seed(_RANDOM_SEED)
    return [20.0 + 5.0 * (i % 10) / 10 + random.gauss(0, noise) for i in range(n)]


class TestSlidingWindowConcurrency:
    """Stress in-memory sliding windows under concurrent append/get."""

    @pytest.mark.parametrize("n_series,ops_per_series", [
        (100, 200),
        (500, 100),
        (1000, 50),
    ])
    def test_concurrent_append_and_get(
        self,
        n_series: int,
        ops_per_series: int,
    ) -> None:
        store = SlidingWindowStore(
            max_size=20,
            max_sensors=n_series + 100,
            ttl_seconds=3600.0,
            enable_proactive_cleanup=False,
        )
        errors: List[str] = []
        latencies: deque = deque(maxlen=n_series * ops_per_series)
        lock = threading.Lock()

        def worker(series_idx: int) -> None:
            sid = _generate_series_id(series_idx)
            base_time = time.monotonic()
            for j in range(ops_per_series):
                t0 = time.monotonic()
                try:
                    r = _make_reading(sid, base_time + j, float(series_idx * 10 + j))
                    store.append(r)
                    win = store.get_window(sid)
                    if not win:
                        with lock:
                            errors.append(f"empty_window:{sid}")
                except Exception as exc:
                    with lock:
                        errors.append(f"{sid}:{type(exc).__name__}:{exc}")
                finally:
                    latencies.append(time.monotonic() - t0)

        t_start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            list(pool.map(worker, range(n_series)))
        total_time = time.monotonic() - t_start

        total_ops = n_series * ops_per_series
        error_rate = len(errors) / total_ops if total_ops else 0.0
        sorted_lat = sorted(latencies)
        p99 = sorted_lat[int(len(sorted_lat) * 0.99)] if sorted_lat else 0.0
        throughput = total_ops / total_time if total_time > 0 else 0.0

        assert error_rate < 0.01, f"error_rate={error_rate:.4f} errors={errors[:5]}"
        assert p99 < 0.5, f"p99_latency={p99:.3f}s exceeds 500ms"
        assert throughput > 100, f"throughput={throughput:.1f} ops/s too low"

        metrics = store.get_metrics()
        assert metrics["active_sensors"] <= n_series
        store.close()


class TestBayesianWeightTrackerConcurrency:
    """Stress BayesianWeightTracker concurrent updates."""

    def test_100_concurrent_updates(self) -> None:
        self._run_tracker_stress(n_threads=100, updates_per_thread=50)

    def test_500_concurrent_updates(self) -> None:
        self._run_tracker_stress(n_threads=500, updates_per_thread=20)

    def test_1000_concurrent_updates(self) -> None:
        self._run_tracker_stress(n_threads=1000, updates_per_thread=10)

    def _run_tracker_stress(self, n_threads: int, updates_per_thread: int) -> None:
        tracker = BayesianWeightTracker(alpha=0.15, max_regimes=5)
        errors: List[str] = []
        lock = threading.Lock()

        def worker(tid: int) -> None:
            regime = f"regime_{tid % 4}"
            engine = f"engine_{tid % 3}"
            for i in range(updates_per_thread):
                try:
                    tracker.update(regime, engine, prediction_error=0.1 + (i % 5) * 0.05)
                except Exception as exc:
                    with lock:
                        errors.append(f"{tid}:{type(exc).__name__}:{exc}")

        t0 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
            list(pool.map(worker, range(n_threads)))
        elapsed = time.monotonic() - t0

        total = n_threads * updates_per_thread
        error_rate = len(errors) / total if total else 0.0
        assert error_rate < 0.01, f"error_rate={error_rate:.4f} errors={errors[:5]}"
        assert elapsed < 30.0, f"elapsed={elapsed:.1f}s too slow for {total} updates"


class TestAnomalyDetectorConcurrency:
    """Stress VotingAnomalyDetector under concurrent detect calls."""

    @pytest.mark.parametrize("n_threads,windows_per_thread", [
        (100, 20),
        (500, 10),
    ])
    def test_concurrent_detect(
        self,
        n_threads: int,
        windows_per_thread: int,
    ) -> None:
        from iot_machine_learning.domain.entities.sensor_reading import SensorWindow

        values = _synthetic_values(100)
        detector = VotingAnomalyDetector(
            config=AnomalyDetectorConfig(min_training_points=20),
        )
        detector.train(values)

        errors: List[str] = []
        lock = threading.Lock()

        def worker(tid: int) -> None:
            for i in range(windows_per_thread):
                try:
                    win_values = values[-20:]
                    readings = [
                        Reading(series_id=str(tid), value=v, timestamp=float(j))
                        for j, v in enumerate(win_values)
                    ]
                    win = SensorWindow(series_id=str(tid), readings=readings)
                    result = detector.detect(win)
                    assert 0.0 <= result.score <= 1.0
                except Exception as exc:
                    with lock:
                        errors.append(f"{tid}:{i}:{type(exc).__name__}:{exc}")

        t0 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            list(pool.map(worker, range(n_threads)))
        elapsed = time.monotonic() - t0

        total = n_threads * windows_per_thread
        error_rate = len(errors) / total if total else 0.0
        assert error_rate < 0.01, f"error_rate={error_rate:.4f} errors={errors[:5]}"
        assert elapsed < 60.0, f"elapsed={elapsed:.1f}s too slow"


class TestWeightedFusionConcurrency:
    """Stress WeightedFusion under concurrent fuse calls."""

    def test_1000_parallel_fusions(self) -> None:
        from iot_machine_learning.infrastructure.ml.cognitive.analysis.types import (
            EnginePerception, InhibitionState,
        )

        fusion = WeightedFusion()
        perceptions = [
            EnginePerception(engine_name="taylor", predicted_value=22.0, confidence=0.8, trend="up"),
            EnginePerception(engine_name="statistical", predicted_value=21.5, confidence=0.7, trend="up"),
            EnginePerception(engine_name="baseline", predicted_value=20.0, confidence=0.9, trend="stable"),
        ]
        inhibitions = [
            InhibitionState(engine_name="taylor", base_weight=0.4, inhibited_weight=0.35, suppression_factor=0.05),
            InhibitionState(engine_name="statistical", base_weight=0.3, inhibited_weight=0.30, suppression_factor=0.0),
            InhibitionState(engine_name="baseline", base_weight=0.3, inhibited_weight=0.30, suppression_factor=0.0),
        ]

        results: List[Tuple] = []
        lock = threading.Lock()

        def worker(_: int) -> None:
            try:
                result = fusion.fuse(perceptions, inhibitions)
                with lock:
                    results.append(result)
            except Exception as exc:
                with lock:
                    results.append(exc)

        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
            list(pool.map(worker, range(1000)))

        errors = [r for r in results if isinstance(r, Exception)]
        assert not errors, f"fusion errors: {errors[:3]}"

        # All weights must sum to 1.0 (within floating point tolerance)
        for r in results:
            weights = r[3]  # type: ignore[index]
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-6, f"weights_sum={total}"


class TestMemoryStability:
    """Ensure memory usage stays bounded under high load."""

    def test_sliding_window_memory_bounded(self) -> None:
        tracemalloc.start()
        store = SlidingWindowStore(max_size=20, max_sensors=1000, enable_proactive_cleanup=False)

        # Warm-up
        for i in range(1000):
            store.append(_make_reading(_generate_series_id(i), float(i), float(i)))

        snap1 = tracemalloc.take_snapshot()

        # Heavy load
        for _ in range(5):
            for i in range(1000):
                store.append(_make_reading(_generate_series_id(i), float(i), float(i)))

        snap2 = tracemalloc.take_snapshot()
        diff = snap2.compare_to(snap1, "lineno")
        total_growth = sum(stat.size_diff for stat in diff if stat.size_diff > 0)

        # With bounded windows, growth should be minimal after warm-up
        assert total_growth < 5 * 1024 * 1024, f"memory_growth={total_growth / 1024 / 1024:.2f}MB"

        store.close()
        tracemalloc.stop()
