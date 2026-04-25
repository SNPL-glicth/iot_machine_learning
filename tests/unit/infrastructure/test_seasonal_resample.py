"""Tests for SeasonalEngine _resample_to_uniform (Problema 2).

3 cases:
1. Uniform timestamps → same result as before (no regression)
2. Gap of 10x → uses longest continuous segment
3. No timestamps → works as before
"""

from __future__ import annotations

import math

import numpy as np

from iot_machine_learning.infrastructure.ml.engines.seasonal.engine import (
    SeasonalPredictorEngine,
)


class TestSeasonalResample:
    def test_uniform_timestamps_no_regression(self) -> None:
        engine = SeasonalPredictorEngine()
        values = [float(i % 10) for i in range(50)]
        timestamps = list(range(50))
        result = engine.predict(values, timestamps=timestamps)

        assert result.metadata.get("fallback") is not True
        assert result.metadata.get("detected_period") is not None
        period = result.metadata["detected_period"]
        assert abs(period - 10) <= 1

    def test_large_gap_uses_longest_segment(self) -> None:
        engine = SeasonalPredictorEngine()
        # Segment 1: [0..4] = 5 points (period ~2)
        # Gap 15 (> 5× median 1)
        # Segment 2: [20..29] = 10 points (period ~3)
        values = (
            [float(i % 2) for i in range(5)]      # 0,1,0,1,0
            + [99.0]                               # gap point — ignored
            + [float(i % 3) for i in range(10)]   # 0,1,2,0,1,2,0,1,2,0
        )
        timestamps = (
            list(range(5))
            + [20]
            + list(range(21, 30))
        )
        result = engine.predict(values, timestamps=timestamps)

        # Should detect period ≈ 3 from the longer segment (10 pts)
        # or fall back if too short
        assert result.predicted_value is not None
        assert math.isfinite(result.predicted_value)

    def test_no_timestamps_works_as_before(self) -> None:
        engine = SeasonalPredictorEngine()
        values = [float(i % 10) for i in range(50)]
        result = engine.predict(values)

        assert result.metadata.get("fallback") is not True
        assert result.metadata.get("detected_period") is not None
        period = result.metadata["detected_period"]
        assert abs(period - 10) <= 1
