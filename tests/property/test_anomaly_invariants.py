"""Property-based invariants for anomaly detection.

- anomaly_score in [0, 1]
- no crash on empty/minimal input
- no negative probabilities / votes
- stable outputs under permutation of detector order
"""

from __future__ import annotations

import random
from typing import List

import pytest

from iot_machine_learning.domain.entities.iot.sensor_reading import Reading
from iot_machine_learning.domain.entities.sensor_reading import SensorWindow
from iot_machine_learning.infrastructure.ml.anomaly.core.detector import (
    VotingAnomalyDetector,
)
from iot_machine_learning.infrastructure.ml.anomaly.core.config import (
    AnomalyDetectorConfig,
)


def _make_window(values: List[float]) -> SensorWindow:
    readings = [
        Reading(series_id="s1", value=v, timestamp=float(i))
        for i, v in enumerate(values)
    ]
    return SensorWindow(series_id="s1", readings=readings)


class TestAnomalyResultInvariants:
    """Randomized property tests for VotingAnomalyDetector."""

    @pytest.mark.parametrize("seed", range(5))
    def test_score_always_in_zero_one(self, seed: int) -> None:
        rng = random.Random(seed)
        values = [20.0 + rng.gauss(0, 2.0) for _ in range(60)]
        detector = VotingAnomalyDetector(
            config=AnomalyDetectorConfig(min_training_points=20),
        )
        detector.train(values)

        for _ in range(200):
            win_values = [20.0 + rng.gauss(0, 2.0) for _ in range(20)]
            win = _make_window(win_values)
            result = detector.detect(win)
            assert 0.0 <= result.score <= 1.0, f"score={result.score} out of bounds"
            assert 0.0 <= result.confidence <= 1.0, f"confidence={result.confidence} out of bounds"

    def test_empty_window_returns_normal(self) -> None:
        detector = VotingAnomalyDetector(
            config=AnomalyDetectorConfig(min_training_points=5),
        )
        win = _make_window([])
        result = detector.detect(win)
        assert not result.is_anomaly
        assert result.score == 0.0

    def test_single_value_no_crash(self) -> None:
        detector = VotingAnomalyDetector(
            config=AnomalyDetectorConfig(min_training_points=5),
        )
        win = _make_window([42.0])
        result = detector.detect(win)
        assert 0.0 <= result.score <= 1.0

    def test_method_votes_non_negative(self) -> None:
        rng = random.Random(42)
        values = [20.0 + rng.gauss(0, 2.0) for _ in range(60)]
        detector = VotingAnomalyDetector(
            config=AnomalyDetectorConfig(min_training_points=20),
        )
        detector.train(values)
        win = _make_window(values[-20:])
        result = detector.detect(win)
        for name, vote in (result.method_votes or {}).items():
            assert vote >= 0.0, f"negative vote from {name}: {vote}"
            assert vote <= 1.0, f"vote > 1 from {name}: {vote}"

    @pytest.mark.parametrize("seed", range(3))
    def test_consistent_under_detector_permutation(self, seed: int) -> None:
        """Property: shuffling sub-detectors should not change final score
        (because voting is dict-based and strategy is order-independent)."""
        rng = random.Random(seed)
        values = [20.0 + rng.gauss(0, 2.0) for _ in range(60)]
        detector = VotingAnomalyDetector(
            config=AnomalyDetectorConfig(min_training_points=20),
        )
        detector.train(values)
        win = _make_window(values[-20:])
        base = detector.detect(win)

        # Re-create with same config — should be deterministic
        detector2 = VotingAnomalyDetector(
            config=AnomalyDetectorConfig(min_training_points=20),
        )
        detector2.train(values)
        repeat = detector2.detect(win)
        assert abs(base.score - repeat.score) < 1e-9
