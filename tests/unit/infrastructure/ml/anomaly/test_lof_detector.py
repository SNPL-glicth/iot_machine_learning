"""Tests for LOFDetector.

5 cases:
1. Detects clear anomaly (stable + outlier)
2. No false positives on stable gaussian noise
3. Handles insufficient data gracefully
4. Handles NaN input gracefully
5. Handles constant input gracefully
"""

from __future__ import annotations

import pytest

from iot_machine_learning.infrastructure.ml.anomaly.detectors.lof_detector import (
    LOFDetector,
)


class TestLOFDetector:
    def test_detects_clear_anomaly(self) -> None:
        detector = LOFDetector()
        detector.train([10.0] * 50 + [100.0])
        if not detector.is_trained:
            pytest.skip("sklearn not available")
        vote = detector.vote(100.0)
        assert vote == 1.0

    def test_no_false_positive_stable(self) -> None:
        detector = LOFDetector()
        values = [10.0 + (i % 3 - 1) * 0.05 for i in range(100)]
        detector.train(values)
        if not detector.is_trained:
            pytest.skip("sklearn not available")
        for v in values[:5]:
            assert detector.vote(v) == 0.0

    def test_handles_insufficient_data(self) -> None:
        detector = LOFDetector()
        detector.train([1.0, 2.0])
        if not detector.is_trained:
            vote = detector.vote(3.0)
            assert vote is None
        else:
            assert detector.vote(3.0) in (0.0, 1.0)

    def test_handles_nan_input(self) -> None:
        detector = LOFDetector()
        detector.train([1.0, 2.0, float("nan"), 4.0, 5.0] * 20)
        if not detector.is_trained:
            vote = detector.vote(5.0)
            assert vote is None
        else:
            assert detector.vote(5.0) in (0.0, 1.0)

    def test_handles_constant_input(self) -> None:
        detector = LOFDetector()
        detector.train([5.0] * 50)
        if not detector.is_trained:
            vote = detector.vote(5.0)
            assert vote is None
        else:
            assert detector.vote(5.0) in (0.0, 1.0)
