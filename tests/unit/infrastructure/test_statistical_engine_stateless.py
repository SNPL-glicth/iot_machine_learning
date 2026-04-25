"""Tests for StatisticalEngine stateless predict() (Problema 3 fix).

3 cases:
1. predict() does not mutate _needs_reoptimization nor _prediction_count
2. record_actual() sets flag after 50 records but does not call _reoptimize
3. optimize() triggers _reoptimize() when flag is active
"""

from __future__ import annotations

import pytest

from iot_machine_learning.infrastructure.ml.engines.statistical.engine import (
    StatisticalPredictionEngine,
)


class TestStatisticalEngineStateless:
    def test_predict_is_stateless(self) -> None:
        engine = StatisticalPredictionEngine(
            enable_optimization=True,
            series_id="test_series",
        )
        initial_flag = engine._needs_reoptimization
        initial_count = engine._prediction_count

        engine.predict([10.0, 11.0, 12.0])

        assert engine._needs_reoptimization == initial_flag
        assert engine._prediction_count == initial_count

    def test_record_actual_defers_reoptimization(self) -> None:
        engine = StatisticalPredictionEngine(
            enable_optimization=True,
            series_id="test_series",
        )
        for i in range(49):
            engine.record_actual(10.0, 10.0 + i * 0.01)

        assert engine._needs_reoptimization is False

        # 50th record should set the flag
        engine.record_actual(10.0, 10.5)
        assert engine._needs_reoptimization is True
        assert engine._prediction_count == 50

    def test_optimize_triggers_reoptimization(self) -> None:
        engine = StatisticalPredictionEngine(
            enable_optimization=True,
            series_id="test_series",
        )
        for i in range(50):
            engine.record_actual(10.0, 10.0 + i * 0.01)

        assert engine._needs_reoptimization is True

        # optimize() should consume the flag
        engine.optimize()
        assert engine._needs_reoptimization is False
        assert engine._prediction_count == 0
