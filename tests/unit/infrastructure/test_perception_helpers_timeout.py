"""Tests for IMP-2 additions: per-engine timeout + engine_failures.

Extends the behaviour covered by ``test_parallel_engine_execution.py``
with scenarios specific to the new timeout kill-switch and the
``consume_engine_failures()`` surface.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from iot_machine_learning.infrastructure.ml.cognitive.perception import helpers


class _MockEngine:
    def __init__(self, name: str, delay_ms: int = 0, should_fail: bool = False) -> None:
        self.name = name
        self.delay_ms = delay_ms
        self.should_fail = should_fail

    def can_handle(self, n_points: int) -> bool:
        return n_points >= 3

    def predict(self, values, timestamps):
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)
        if self.should_fail:
            raise RuntimeError(f"Engine {self.name} failed")
        mock = MagicMock()
        mock.predicted_value = sum(values) / len(values)
        mock.confidence = 0.8
        mock.trend = "stable"
        mock.metadata = {"diagnostic": {"stability_indicator": 0.5}}
        return mock


class TestEngineTimeout:
    def test_slow_engine_times_out(self) -> None:
        """Engine exceeding ML_PREDICT_ENGINE_TIMEOUT_MS is dropped."""
        engines = [
            _MockEngine("fast1", delay_ms=10),
            _MockEngine("slow", delay_ms=400),  # exceeds 50ms budget below
            _MockEngine("fast2", delay_ms=10),
        ]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        result = helpers._collect_perceptions_parallel(
            engines, values, None, max_workers=3, timeout_ms=50.0
        )

        names = {p.engine_name for p in result}
        assert "fast1" in names
        assert "fast2" in names
        assert "slow" not in names

    def test_timeout_recorded_in_failures(self) -> None:
        """Timed-out engine surfaces in consume_engine_failures()."""
        engines = [
            _MockEngine("fast", delay_ms=10),
            _MockEngine("slow", delay_ms=300),
        ]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        # Force parallel by bumping workers > 1
        with patch.object(helpers, "ML_PREDICT_MAX_WORKERS", 2), \
             patch.object(helpers, "ML_PREDICT_ENGINE_TIMEOUT_MS", 50.0):
            helpers.collect_perceptions(engines, values, None)
            failures = helpers.consume_engine_failures()

        reasons = {(f["engine"], f["reason"]) for f in failures}
        assert ("slow", "timeout") in reasons

    def test_all_fast_no_failures(self) -> None:
        engines = [_MockEngine("a"), _MockEngine("b"), _MockEngine("c")]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        with patch.object(helpers, "ML_PREDICT_MAX_WORKERS", 3):
            helpers.collect_perceptions(engines, values, None)
            failures = helpers.consume_engine_failures()
        assert failures == []


class TestEngineFailures:
    def test_exception_recorded(self) -> None:
        engines = [
            _MockEngine("a"),
            _MockEngine("b", should_fail=True),
            _MockEngine("c"),
        ]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        with patch.object(helpers, "ML_PREDICT_MAX_WORKERS", 3):
            result = helpers.collect_perceptions(engines, values, None)
            failures = helpers.consume_engine_failures()
        assert len(result) == 2
        assert {(f["engine"], f["reason"]) for f in failures} == {("b", "exception")}
        # Error message captured
        f_b = next(f for f in failures if f["engine"] == "b")
        assert "Engine b failed" in f_b["error"]

    def test_cannot_handle_recorded(self) -> None:
        class _TooSmallEngine(_MockEngine):
            def can_handle(self, n_points: int) -> bool:
                return n_points >= 1000  # always false for typical input

        engines = [_MockEngine("a"), _TooSmallEngine("big")]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        helpers.collect_perceptions(engines, values, None)
        failures = helpers.consume_engine_failures()
        reasons = {(f["engine"], f["reason"]) for f in failures}
        assert ("big", "cannot_handle") in reasons

    def test_consume_clears_buffer(self) -> None:
        engines = [_MockEngine("a"), _MockEngine("b", should_fail=True)]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        helpers.collect_perceptions(engines, values, None)
        first = helpers.consume_engine_failures()
        second = helpers.consume_engine_failures()
        assert len(first) >= 1
        assert second == []

    def test_sequential_path_also_records_failures(self) -> None:
        engines = [
            _MockEngine("a"),
            _MockEngine("b", should_fail=True),
        ]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        with patch.object(helpers, "ML_PREDICT_MAX_WORKERS", 1):
            helpers.collect_perceptions(engines, values, None)
            failures = helpers.consume_engine_failures()
        assert any(f["engine"] == "b" and f["reason"] == "exception" for f in failures)


class TestSequentialTimeout:
    def test_sequential_timeout_drops_slow_engine(self) -> None:
        """Engine sleeping 600 ms is dropped when ML_PREDICT_MAX_WORKERS=1."""
        engines = [
            _MockEngine("fast", delay_ms=10),
            _MockEngine("slow", delay_ms=600),
        ]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        with patch.object(helpers, "ML_PREDICT_MAX_WORKERS", 1), \
             patch.object(helpers, "ML_PREDICT_ENGINE_TIMEOUT_MS", 50.0):
            result = helpers.collect_perceptions(engines, values, None)
            failures = helpers.consume_engine_failures()

        names = {p.engine_name for p in result}
        assert "fast" in names
        assert "slow" not in names

        reasons = {(f["engine"], f["reason"]) for f in failures}
        assert ("slow", "timeout") in reasons

    def test_sequential_timeout_other_engines_still_run(self) -> None:
        """One engine times out; two others complete normally."""
        engines = [
            _MockEngine("fast_a", delay_ms=10),
            _MockEngine("slow", delay_ms=600),
            _MockEngine("fast_b", delay_ms=10),
        ]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        with patch.object(helpers, "ML_PREDICT_MAX_WORKERS", 1), \
             patch.object(helpers, "ML_PREDICT_ENGINE_TIMEOUT_MS", 50.0):
            result = helpers.collect_perceptions(engines, values, None)
            failures = helpers.consume_engine_failures()

        names = {p.engine_name for p in result}
        assert "fast_a" in names
        assert "fast_b" in names
        assert "slow" not in names
        assert len(result) == 2

        reasons = {(f["engine"], f["reason"]) for f in failures}
        assert ("slow", "timeout") in reasons


class TestEnvOverrides:
    def test_timeout_env_default(self) -> None:
        # Default is 400.0 per IMP-2 spec answer §2.
        assert helpers.ML_PREDICT_ENGINE_TIMEOUT_MS == 400.0

    def test_max_workers_default(self) -> None:
        assert helpers.ML_PREDICT_MAX_WORKERS == 3
