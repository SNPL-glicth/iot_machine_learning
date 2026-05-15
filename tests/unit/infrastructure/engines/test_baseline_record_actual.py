"""Tests para BaselineMovingAverageEngine.record_actual (P1).

Verifica que el motor baseline pueda participar en el ciclo de
feedback del orchestrator (BayesianWeightTracker / error tracking).
"""

from __future__ import annotations

import pytest

from iot_machine_learning.infrastructure.ml.engines.core.factory import (
    BaselineMovingAverageEngine,
)


class TestBaselineRecordActual:
    """P1: record_actual() transversal para BaselineMovingAverageEngine."""

    def test_record_actual_does_not_crash(self) -> None:
        """Calling record_actual after predict should be safe."""
        engine = BaselineMovingAverageEngine()
        result = engine.predict([1.0, 2.0, 3.0])
        engine.record_actual(result.predicted_value, 2.5)

    def test_error_history_accumulates(self) -> None:
        """Each call appends one entry to _error_history."""
        engine = BaselineMovingAverageEngine()
        engine.record_actual(10.0, 12.0)
        engine.record_actual(10.0, 8.0)
        assert len(engine._error_history) == 2
        assert list(engine._error_history) == [2.0, 2.0]

    def test_recent_mae_computed_correctly(self) -> None:
        """recent_mae returns mean of stored absolute errors."""
        engine = BaselineMovingAverageEngine()
        engine.record_actual(10.0, 13.0)  # error 3.0
        engine.record_actual(10.0, 11.0)  # error 1.0
        engine.record_actual(10.0, 10.0)  # error 0.0
        assert engine.recent_mae() == pytest.approx(4.0 / 3, abs=1e-6)

    def test_recent_mae_returns_none_when_empty(self) -> None:
        """Before any record_actual, recent_mae is None."""
        engine = BaselineMovingAverageEngine()
        assert engine.recent_mae() is None

    def test_history_bounded_to_50(self) -> None:
        """Deque maxlen=50 prevents unbounded growth."""
        engine = BaselineMovingAverageEngine()
        for i in range(60):
            engine.record_actual(float(i), float(i) + 1.0)
        assert len(engine._error_history) == 50
        # error is always abs(predicted - actual) = abs(i - (i+1)) = 1.0
        assert engine._error_history[0] == 1.0
        assert engine._error_history[-1] == 1.0

    def test_record_actual_with_invalid_inputs(self) -> None:
        """NaN / inf in predicted or actual should still store error."""
        engine = BaselineMovingAverageEngine()
        engine.record_actual(float("nan"), 5.0)
        assert len(engine._error_history) == 1
        assert engine._error_history[0] != engine._error_history[0]  # NaN

    def test_full_predict_record_cycle(self) -> None:
        """End-to-end: predict then record actual, then check mae."""
        engine = BaselineMovingAverageEngine(window=3)
        result = engine.predict([10.0, 20.0, 30.0])
        # predicted should be 20.0 (mean of last 3)
        assert result.predicted_value == pytest.approx(20.0, abs=1e-6)
        engine.record_actual(result.predicted_value, 25.0)
        assert engine.recent_mae() == pytest.approx(5.0, abs=1e-6)
