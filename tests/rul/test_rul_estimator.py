"""Tests for RUL estimator core logic."""

from __future__ import annotations

import pytest

from iot_machine_learning.infrastructure.ml.anomaly.rul.estimator import RULEstimator


class TestRULEstimator:

    def test_no_deterioration(self):
        estimator = RULEstimator()
        result = estimator.estimate(
            anomaly_score=0.0,
            drift_magnitude=0.0,
            consecutive_anomalies=0,
        )
        assert result is None

    def test_critical_urgency(self):
        estimator = RULEstimator()
        result = estimator.estimate(
            anomaly_score=0.9,
            drift_magnitude=0.8,
            consecutive_anomalies=15,
        )
        assert result is not None
        assert result.urgency == RULEstimator.URGENCY_CRITICAL
        assert result.time_to_failure_hours < 4.0

    def test_medium_urgency(self):
        estimator = RULEstimator()
        result = estimator.estimate(
            anomaly_score=0.5,
            drift_magnitude=0.3,
            consecutive_anomalies=7,
        )
        assert result is not None
        assert result.urgency == RULEstimator.URGENCY_MEDIUM
        assert 4.0 <= result.time_to_failure_hours <= 24.0

    def test_low_urgency(self):
        estimator = RULEstimator()
        result = estimator.estimate(
            anomaly_score=0.2,
            drift_magnitude=0.1,
            consecutive_anomalies=3,
        )
        assert result is not None
        assert result.urgency == RULEstimator.URGENCY_LOW
        assert result.time_to_failure_hours > 24.0

    def test_confidence_levels(self):
        estimator = RULEstimator()

        low = estimator.estimate(
            anomaly_score=0.5,
            drift_magnitude=0.3,
            consecutive_anomalies=3,
        )
        assert low is not None
        assert low.confidence == RULEstimator.CONFIDENCE_LOW

        medium = estimator.estimate(
            anomaly_score=0.5,
            drift_magnitude=0.3,
            consecutive_anomalies=6,
        )
        assert medium is not None
        assert medium.confidence == RULEstimator.CONFIDENCE_MEDIUM

        high = estimator.estimate(
            anomaly_score=0.5,
            drift_magnitude=0.3,
            consecutive_anomalies=11,
        )
        assert high is not None
        assert high.confidence == RULEstimator.CONFIDENCE_HIGH

    def test_clamp_bounds(self):
        estimator = RULEstimator()

        max_rate = estimator.estimate(
            anomaly_score=1.0,
            drift_magnitude=1.0,
            consecutive_anomalies=5,
        )
        assert max_rate is not None
        assert max_rate.time_to_failure_hours >= 0.5

        min_rate = estimator.estimate(
            anomaly_score=0.001,
            drift_magnitude=0.001,
            consecutive_anomalies=5,
        )
        assert min_rate is not None
        assert min_rate.time_to_failure_hours <= 168.0
