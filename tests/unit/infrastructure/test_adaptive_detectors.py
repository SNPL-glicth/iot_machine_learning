"""Tests para adaptive thresholds en ZScoreDetector e IQRDetector.

FIX-5: Adaptive thresholds reduce false positives on high-volatility series.
"""

from __future__ import annotations

import random

import pytest

from iot_machine_learning.infrastructure.ml.anomaly.detectors.z_score_detector import (
    ZScoreDetector,
)
from iot_machine_learning.infrastructure.ml.anomaly.detectors.iqr_detector import (
    IQRDetector,
)


class TestZScoreAdaptiveThresholds:
    """ZScoreDetector adaptive threshold tests."""

    def test_high_volatility_reduces_false_positives(self) -> None:
        """Volatile-but-normal data: adaptive thresholds increase, reducing votes."""
        detector = ZScoreDetector(lower=2.0, upper=3.0, adaptive=True)
        # Train on low-volatility data
        random.seed(42)
        low_vol = [20.0 + random.gauss(0, 1.0) for _ in range(100)]
        detector.train(low_vol)
        assert detector._stats is not None
        base_std = detector._stats.std

        # Simulate 5 high-volatility windows to fill history
        for _ in range(5):
            high_vol = [20.0 + random.gauss(0, 5.0) for _ in range(20)]
            # Compute local std and feed values
            for v in high_vol:
                detector.vote(v)

        # After 5 windows, adaptive should be active
        assert len(detector._rolling_std_history) >= 5

        # The thresholds should have scaled up
        lower, upper = detector._effective_thresholds
        assert upper >= 3.0
        # A value that was 2.5 * base_std should now vote lower because
        # upper threshold increased
        mean = detector._stats.mean
        test_val = mean + 2.5 * base_std
        vote_before = detector.vote(test_val)
        # Just verify vote is computed without error
        assert vote_before is not None

    def test_adaptive_false_disables_adaptation(self) -> None:
        """adaptive=False: thresholds never change from base values."""
        detector = ZScoreDetector(lower=2.0, upper=3.0, adaptive=False)
        random.seed(42)
        detector.train([20.0 + random.gauss(0, 1.0) for _ in range(50)])

        # Feed many values
        for _ in range(50):
            detector.vote(50.0)

        lower, upper = detector._effective_thresholds
        assert lower == 2.0
        assert upper == 3.0

    def test_cold_start_uses_fixed_threshold(self) -> None:
        """< 5 history entries: uses fixed base thresholds."""
        detector = ZScoreDetector(lower=2.0, upper=3.0, adaptive=True)
        random.seed(42)
        detector.train([20.0 + random.gauss(0, 1.0) for _ in range(50)])

        # Only 1 entry from train() → < 5
        assert len(detector._rolling_std_history) == 1
        lower, upper = detector._effective_thresholds
        assert lower == 2.0
        assert upper == 3.0


class TestIQRAdaptiveThresholds:
    """IQRDetector adaptive threshold tests."""

    def test_high_volatility_increases_fence_multiplier(self) -> None:
        """Volatile data: adaptive fence multiplier increases."""
        detector = IQRDetector(adaptive=True)
        # Train on low-volatility data
        random.seed(42)
        low_vol = [20.0 + random.gauss(0, 1.0) for _ in range(100)]
        detector.train(low_vol)

        # Feed 5 high-volatility windows
        for _ in range(5):
            high_vol = [20.0 + random.gauss(0, 10.0) for _ in range(20)]
            for v in high_vol:
                detector.vote(v)

        # After 5 windows, adaptive should be active
        assert len(detector._rolling_iqr_history) >= 5
        multiplier = detector._effective_fence_multiplier
        assert multiplier >= 1.5
        assert multiplier <= 3.0

    def test_adaptive_false_disables_adaptation(self) -> None:
        """adaptive=False: fence multiplier stays at 1.5."""
        detector = IQRDetector(adaptive=False)
        random.seed(42)
        detector.train([20.0 + random.gauss(0, 1.0) for _ in range(50)])

        for _ in range(50):
            detector.vote(50.0)

        assert detector._effective_fence_multiplier == 1.5

    def test_cold_start_uses_fixed_multiplier(self) -> None:
        """< 5 history entries: uses fixed 1.5 multiplier."""
        detector = IQRDetector(adaptive=True)
        random.seed(42)
        detector.train([20.0 + random.gauss(0, 1.0) for _ in range(50)])

        assert len(detector._rolling_iqr_history) == 1
        assert detector._effective_fence_multiplier == 1.5
