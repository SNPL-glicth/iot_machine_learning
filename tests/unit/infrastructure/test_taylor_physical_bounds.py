"""Tests para TaylorEngine physical bounds (FIX-9).

Physical min/max clamp after observed-range clamping.
"""

from __future__ import annotations

from iot_machine_learning.infrastructure.ml.engines.taylor.engine import (
    TaylorPredictionEngine,
)


class TestTaylorPhysicalBounds:
    """Physical bounds awareness."""

    def test_physical_max_clamps_extrapolation(self) -> None:
        """Rising series extrapolates above physical_max → clamped."""
        engine = TaylorPredictionEngine(
            order=1,
            physical_max=30.0,
        )
        # Steeply rising series: predicted will be >> 30
        values = [10.0, 15.0, 20.0, 25.0, 28.0]
        result = engine.predict(values)
        assert result.predicted_value <= 30.0
        assert result.metadata["physical_clamp_applied"] is True
        assert result.metadata["physical_clamp_direction"] == "max"

    def test_physical_min_clamps_extrapolation(self) -> None:
        """Falling series extrapolates below physical_min → clamped."""
        engine = TaylorPredictionEngine(
            order=1,
            physical_min=0.0,
        )
        # Steeply falling series: predicted will be < 0
        values = [20.0, 15.0, 10.0, 5.0, 2.0]
        result = engine.predict(values)
        assert result.predicted_value >= 0.0
        assert result.metadata["physical_clamp_applied"] is True
        assert result.metadata["physical_clamp_direction"] == "min"

    def test_physical_bounds_none_no_change(self) -> None:
        """Default None → exact old behavior, no regression."""
        engine = TaylorPredictionEngine(order=1)
        values = [10.0, 15.0, 20.0, 25.0, 28.0]
        result = engine.predict(values)
        assert result.metadata["physical_clamp_applied"] is False
        assert result.metadata["physical_clamp_direction"] is None
