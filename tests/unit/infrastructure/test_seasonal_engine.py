"""Tests para SeasonalPredictorEngine (FIX-12).

NaN guard + basic behavior.
"""

from __future__ import annotations

import math

import numpy as np

from iot_machine_learning.infrastructure.ml.engines.seasonal.engine import (
    SeasonalConfig,
    SeasonalPredictorEngine,
)


class TestSeasonalEngine:
    """Seasonal FFT-based prediction tests."""

    def test_predict_basic_seasonal_pattern(self) -> None:
        """50 values with known period=10 → detected period ≈ 10."""
        engine = SeasonalPredictorEngine()
        values = [float(i % 10) for i in range(50)]
        result = engine.predict(values)

        assert result.metadata.get("fallback") is not True
        assert result.metadata.get("detected_period") is not None
        period = result.metadata["detected_period"]
        assert abs(period - 10) <= 1

    def test_predict_constant_input_returns_fallback(self) -> None:
        """All same value → no crash, returns fallback."""
        engine = SeasonalPredictorEngine()
        values = [5.0] * 20
        result = engine.predict(values)

        assert result.metadata.get("fallback") is True

    def test_predict_nan_input_filtered(self) -> None:
        """Input with NaN values → filters and continues or fallback."""
        engine = SeasonalPredictorEngine()
        values = [1.0, 2.0, float("nan"), 4.0] * 3
        result = engine.predict(values)

        # Either detects or falls back gracefully — no exception
        assert result.predicted_value is not None
        assert math.isfinite(result.predicted_value)

    def test_predict_insufficient_data_returns_fallback(self) -> None:
        """3 values < min_period*2 → fallback."""
        engine = SeasonalPredictorEngine()
        values = [1.0, 2.0, 3.0]
        result = engine.predict(values)

        assert result.metadata.get("fallback") is True

    def test_predict_single_outlier_does_not_crash(self) -> None:
        """One extreme value in otherwise normal series → no exception."""
        engine = SeasonalPredictorEngine()
        values = [float(i % 5) for i in range(50)]
        values[25] = 1e9  # Extreme outlier
        result = engine.predict(values)

        assert result.predicted_value is not None
        assert math.isfinite(result.predicted_value)
