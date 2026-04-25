"""Tests para BaselineMovingAverageEngine window fix (FIX-11).
"""

from __future__ import annotations

from iot_machine_learning.infrastructure.ml.engines.core.factory import (
    BaselineMovingAverageEngine,
)


class TestBaselineWindowFix:
    """Window parameter is respected; default is min(20, len(values))."""

    def test_factory_respects_window_kwarg(self) -> None:
        """window=5 passed → uses last 5 values not all values."""
        engine = BaselineMovingAverageEngine(window=5)
        values = [10.0] * 50 + [20.0] * 50  # avg should be 20.0 if last 5
        result = engine.predict(values)

        assert result.metadata["window"] == 5
        assert result.predicted_value == 20.0  # last 5 are all 20.0

    def test_factory_default_window_is_20_not_len_values(self) -> None:
        """100 values, no window kwarg → engine uses last 20."""
        engine = BaselineMovingAverageEngine()
        values = [10.0] * 80 + [20.0] * 20
        result = engine.predict(values)

        assert result.metadata["window"] == 20
        assert result.predicted_value == 20.0

    def test_window_larger_than_data_uses_all_data(self) -> None:
        """Graceful when window > len(values)."""
        engine = BaselineMovingAverageEngine(window=50)
        values = [10.0] * 5
        result = engine.predict(values)

        assert result.metadata["window"] == 50
        assert result.predicted_value == 10.0
