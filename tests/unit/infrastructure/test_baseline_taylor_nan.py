"""Tests for NaN/inf sanitization in Baseline and Taylor engines (Problema 2).

3 cases per engine:
- NaN input → no exception, valid fallback PredictionResult
- inf input → no exception, valid fallback PredictionResult
- Mixed valid/invalid → predicts with clean values only
"""

from __future__ import annotations

import math

import pytest

from iot_machine_learning.infrastructure.ml.engines.core.factory import (
    BaselineMovingAverageEngine,
)
from iot_machine_learning.infrastructure.ml.engines.taylor.engine import (
    TaylorPredictionEngine,
)


class TestBaselineNanInf:
    def test_nan_returns_fallback(self) -> None:
        engine = BaselineMovingAverageEngine()
        result = engine.predict([float("nan"), float("nan")])
        assert result.metadata["reason"] == "all_inputs_invalid"
        assert result.confidence == 0.0

    def test_inf_returns_fallback(self) -> None:
        engine = BaselineMovingAverageEngine()
        result = engine.predict([float("inf"), float("-inf")])
        assert result.metadata["reason"] == "all_inputs_invalid"
        assert result.confidence == 0.0

    def test_mixed_predicts_with_clean_only(self) -> None:
        engine = BaselineMovingAverageEngine()
        result = engine.predict([10.0, float("nan"), 20.0, float("inf"), 30.0])
        # clean values are [10.0, 20.0, 30.0] → avg = 20.0
        assert result.predicted_value == 20.0
        assert result.confidence > 0.0
        assert "reason" not in result.metadata


class TestTaylorNanInf:
    def test_nan_returns_fallback(self) -> None:
        engine = TaylorPredictionEngine()
        result = engine.predict([float("nan"), float("nan"), float("nan")])
        assert result.metadata["reason"] == "all_inputs_invalid"
        assert result.confidence == 0.0

    def test_inf_returns_fallback(self) -> None:
        engine = TaylorPredictionEngine()
        result = engine.predict([float("inf"), float("-inf")])
        assert result.metadata["reason"] == "all_inputs_invalid"
        assert result.confidence == 0.0

    def test_mixed_predicts_with_clean_only(self) -> None:
        engine = TaylorPredictionEngine()
        values = [10.0, float("nan"), 20.0, float("inf"), 30.0, 40.0, 50.0]
        result = engine.predict(values)
        # should not raise, clean values are [10,20,30,40,50]
        assert result.predicted_value is not None
        assert result.confidence > 0.0
        assert "reason" not in result.metadata
