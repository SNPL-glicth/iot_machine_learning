"""Tests for engines/statistical_engine.py — EMA/Holt prediction."""

from __future__ import annotations

import math
import random

import pytest

from iot_machine_learning.infrastructure.ml.engines.statistical_engine import (
    StatisticalPredictionEngine,
)


class TestStatisticalEngineBasic:

    def test_constant_signal(self) -> None:
        eng = StatisticalPredictionEngine()
        result = eng.predict([5.0] * 20)
        assert result.predicted_value == pytest.approx(5.0, abs=0.1)
        assert result.trend == "stable"

    def test_linear_signal(self) -> None:
        eng = StatisticalPredictionEngine(alpha=0.5, beta=0.3)
        values = [float(i) for i in range(20)]
        result = eng.predict(values)
        # Should predict near 20.0
        assert result.predicted_value > 18.0
        assert result.trend == "up"

    def test_downward_trend(self) -> None:
        eng = StatisticalPredictionEngine(alpha=0.5, beta=0.3)
        values = [100.0 - float(i) for i in range(20)]
        result = eng.predict(values)
        assert result.trend == "down"

    def test_confidence_range(self) -> None:
        eng = StatisticalPredictionEngine()
        result = eng.predict([1.0, 2.0, 3.0, 4.0, 5.0])
        assert 0.0 <= result.confidence <= 1.0

    def test_finite_prediction(self) -> None:
        random.seed(42)
        eng = StatisticalPredictionEngine()
        values = [random.gauss(50, 10) for _ in range(30)]
        result = eng.predict(values)
        assert math.isfinite(result.predicted_value)


class TestStatisticalEngineMetadata:

    def test_metadata_keys(self) -> None:
        eng = StatisticalPredictionEngine()
        result = eng.predict([1.0, 2.0, 3.0, 4.0, 5.0])
        m = result.metadata
        assert "level" in m
        assert "trend_component" in m
        assert "alpha" in m
        assert "beta" in m
        assert "residual_std" in m
        assert "diagnostic" in m

    def test_diagnostic_has_stability(self) -> None:
        eng = StatisticalPredictionEngine()
        result = eng.predict([1.0, 2.0, 3.0, 4.0, 5.0])
        diag = result.metadata["diagnostic"]
        assert "stability_indicator" in diag
        assert "local_fit_error" in diag
        assert "method" in diag
        assert diag["method"] == "ema_holt"


class TestStatisticalEngineValidation:

    def test_empty_values_raises(self) -> None:
        eng = StatisticalPredictionEngine()
        with pytest.raises(ValueError):
            eng.predict([])

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError):
            StatisticalPredictionEngine(alpha=0.0)
        with pytest.raises(ValueError):
            StatisticalPredictionEngine(alpha=1.5)

    def test_invalid_beta_raises(self) -> None:
        with pytest.raises(ValueError):
            StatisticalPredictionEngine(beta=-0.1)

    def test_invalid_horizon_raises(self) -> None:
        with pytest.raises(ValueError):
            StatisticalPredictionEngine(horizon=0)

    def test_can_handle(self) -> None:
        eng = StatisticalPredictionEngine()
        assert eng.can_handle(3) is True
        assert eng.can_handle(2) is False

    def test_name(self) -> None:
        eng = StatisticalPredictionEngine()
        assert eng.name == "statistical_ema_holt"

    def test_supports_uncertainty(self) -> None:
        eng = StatisticalPredictionEngine()
        assert eng.supports_uncertainty() is False


class TestStatisticalEngineFallback:

    def test_insufficient_data_fallback(self) -> None:
        eng = StatisticalPredictionEngine()
        result = eng.predict([5.0, 6.0])
        assert result.metadata.get("fallback") == "insufficient_data"
        assert result.predicted_value == pytest.approx(5.5)
