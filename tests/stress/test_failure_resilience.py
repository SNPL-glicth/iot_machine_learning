"""Failure handling & graceful degradation under partial failures.

Ensures:
- fallback predictions when engines fail
- safe defaults when Redis is unavailable
- no system crashes under partial failures
- circuit breaker transitions protect downstream
"""

from __future__ import annotations

import threading
import time
from typing import Any, List

import pytest

from iot_machine_learning.infrastructure.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitState,
)
from iot_machine_learning.infrastructure.ml.cognitive.fusion.engine_selector import (
    WeightedFusion,
)
from iot_machine_learning.infrastructure.ml.cognitive.analysis.types import (
    EnginePerception, InhibitionState,
)
from iot_machine_learning.infrastructure.ml.cognitive.error_store import (
    EngineErrorStore,
)
from iot_machine_learning.domain.entities.iot.sensor_reading import Reading
from iot_machine_learning.domain.entities.sensor_reading import SensorWindow
from iot_machine_learning.infrastructure.ml.anomaly.core.detector import (
    VotingAnomalyDetector,
)
from iot_machine_learning.infrastructure.ml.anomaly.core.config import (
    AnomalyDetectorConfig,
)
from iot_machine_learning.ml_service.consumers.sliding_window import (
    SlidingWindowStore,
)


class TestCircuitBreakerResilience:
    """Circuit breaker must fail-fast and recover gracefully."""

    def test_open_circuit_fails_fast(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.1)
        # First failure: original exception propagates, circuit transitions to OPEN
        with pytest.raises(RuntimeError, match="boom"):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))

        # Immediately after failure, circuit should be open
        with pytest.raises(CircuitBreakerOpen):
            cb.call(lambda: "ok")

    def test_half_open_recovery(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.05, half_open_max_calls=1, success_threshold=1)
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))

        time.sleep(0.08)
        result = cb.call(lambda: "recovered")
        assert result == "recovered"
        assert cb.state == CircuitState.CLOSED

    def test_circuit_metrics_exposed(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=1.0)
        metrics = cb.get_metrics()
        assert metrics["name"] == "test"
        assert metrics["state"] == "CLOSED"


class TestEngineErrorStoreResilience:
    """EngineErrorStore degrades to in-memory when Redis is unavailable."""

    def test_no_redis_never_raises(self) -> None:
        store = EngineErrorStore(redis_client=None)
        for i in range(1000):
            store.record("s1", "engine_a", float(i % 10))

        recent = store.get_recent("s1", "engine_a", 5)
        assert len(recent) == 5
        assert recent == [5.0, 6.0, 7.0, 8.0, 9.0]

    def test_invalid_errors_silently_dropped(self) -> None:
        store = EngineErrorStore(redis_client=None)
        store.record("s1", "engine_a", -1.0)
        store.record("s1", "engine_a", float("nan"))
        store.record("s1", "engine_a", float("inf"))
        recent = store.get_recent("s1", "engine_a", 10)
        assert len(recent) == 0


class TestSlidingWindowGracefulDegradation:
    """SlidingWindowStore must never crash regardless of input."""

    def test_extreme_values_no_crash(self) -> None:
        from iot_machine_learning.domain.entities.iot.sensor_reading import Reading
        store = SlidingWindowStore(max_size=10, enable_proactive_cleanup=False)
        # Reading validates finiteness; store itself must not crash
        valid_extremes = [1e308, -1e308, 0.0]
        for i, v in enumerate(valid_extremes):
            store.append(Reading(series_id="s1", value=v, timestamp=float(i)))
        # inf/nan rejected at Reading level — store never sees them
        store.close()

    def test_rapid_open_close_no_crash(self) -> None:
        for _ in range(100):
            store = SlidingWindowStore(max_size=5, enable_proactive_cleanup=True)
            store.close()


class TestWeightedFusionFallback:
    """Fusion must produce safe defaults under partial engine failure."""

    def test_all_engines_inhibited_returns_uniform_weights(self) -> None:
        fusion = WeightedFusion()
        perceptions = [
            EnginePerception("e1", 20.0, 0.8, "up"),
            EnginePerception("e2", 21.0, 0.7, "up"),
        ]
        inhibitions = [
            InhibitionState(engine_name="e1", base_weight=0.5, inhibited_weight=0.0, suppression_factor=1.0),
            InhibitionState(engine_name="e2", base_weight=0.5, inhibited_weight=0.0, suppression_factor=1.0),
        ]
        value, confidence, trend, weights, selected, reason = fusion.fuse(perceptions, inhibitions)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert value == pytest.approx(20.5, abs=0.5)

    def test_no_perceptions_safe_fallback(self) -> None:
        fusion = WeightedFusion()
        result = fusion.fuse([], [])
        value, confidence, trend, weights, selected, reason = result
        assert value == 0.0
        assert confidence == 0.0
        assert trend == "stable"
        assert reason == "no_engines"


class TestAnomalyDetectorGracefulDegradation:
    """Anomaly detector must not crash on edge-case inputs."""

    def test_all_same_values_no_crash(self) -> None:
        detector = VotingAnomalyDetector(
            config=AnomalyDetectorConfig(min_training_points=10),
        )
        detector.train([50.0] * 100)
        readings = [Reading(series_id="s1", value=50.0, timestamp=float(i)) for i in range(20)]
        win = SensorWindow(series_id="s1", readings=readings)
        result = detector.detect(win)
        assert 0.0 <= result.score <= 1.0

    def test_single_spike_no_crash(self) -> None:
        detector = VotingAnomalyDetector(
            config=AnomalyDetectorConfig(min_training_points=10),
        )
        values = [20.0] * 99 + [1000.0]
        detector.train(values)
        readings = [Reading(series_id="s1", value=v, timestamp=float(i)) for i, v in enumerate(values[-20:])]
        win = SensorWindow(series_id="s1", readings=readings)
        result = detector.detect(win)
        assert 0.0 <= result.score <= 1.0


class TestConcurrentPartialFailure:
    """Partial failures in one thread must not corrupt shared state."""

    def test_error_store_concurrent_mixed_valid_invalid(self) -> None:
        store = EngineErrorStore(redis_client=None)
        errors: List[str] = []
        lock = threading.Lock()

        def worker(tid: int) -> None:
            for i in range(100):
                try:
                    if i % 5 == 0:
                        store.record(f"s{tid}", "e", float("nan"))
                    else:
                        store.record(f"s{tid}", "e", float(i))
                except Exception as exc:
                    with lock:
                        errors.append(str(exc))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"unexpected errors: {errors[:5]}"
        for tid in range(20):
            recent = store.get_recent(f"s{tid}", "e", 10)
            assert len(recent) == 10
            assert all(r >= 0.0 for r in recent)
