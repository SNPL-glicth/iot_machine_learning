"""Tests for per-engine individual timeout (Problema 3).

3 cases:
1. Fast engine (100ms) with 200ms timeout → completes normally
2. Slow engine (500ms) with 200ms timeout → returns None, no exception
3. Engine that raises exception → returns None, does not propagate
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


class TestEngineIndividualTimeout:
    def test_fast_engine_completes_within_timeout(self) -> None:
        engine = _MockEngine("fast", delay_ms=100)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = helpers._run_engine_with_timeout(
            engine, values, None, timeout_ms=200.0
        )
        assert result is not None
        assert result.engine_name == "fast"

    def test_slow_engine_returns_none_on_timeout(self) -> None:
        engine = _MockEngine("slow", delay_ms=500)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = helpers._run_engine_with_timeout(
            engine, values, None, timeout_ms=200.0
        )
        assert result is None

    def test_failing_engine_returns_none_no_exception(self) -> None:
        engine = _MockEngine("failer", should_fail=True)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = helpers._run_engine_with_timeout(
            engine, values, None, timeout_ms=200.0
        )
        assert result is None
