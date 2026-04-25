"""Tests for IQRDetector.

5 cases:
1. Detects clear anomaly (stable + outlier)
2. No false positives on stable gaussian noise
3. Handles insufficient data gracefully
4. Handles NaN input gracefully
5. Handles constant input gracefully
"""

from __future__ import annotations

import math

import pytest

from iot_machine_learning.infrastructure.ml.anomaly.detectors.iqr_detector import (
    IQRDetector,
)


class TestIQRDetector:
    def test_detects_clear_anomaly(self) -> None:
        detector = IQRDetector()
        detector.train([10.0] * 20 + [100.0])
        vote = detector.vote(100.0)
        assert vote == 1.0

    def test_no_false_positive_stable(self) -> None:
        detector = IQRDetector()
        values = [10.0 + (i % 3 - 1) * 0.05 for i in range(50)]
        detector.train(values)
        for v in values:
            assert detector.vote(v) == 0.0

    def test_handles_insufficient_data(self) -> None:
        detector = IQRDetector()
        detector.train([1.0, 2.0])  # < 5 entries for adaptive fences
        vote = detector.vote(3.0)
        assert vote in (0.0, 1.0)  # falls back to fixed multiplier

    def test_handles_nan_input(self) -> None:
        detector = IQRDetector()
        detector.train([1.0, 2.0, float("nan"), 4.0, 5.0] * 10)
        vote = detector.vote(5.0)
        assert vote in (0.0, 1.0)

    def test_handles_constant_input(self) -> None:
        detector = IQRDetector()
        detector.train([5.0] * 50)
        vote = detector.vote(5.0)
        assert vote in (0.0, 1.0)
