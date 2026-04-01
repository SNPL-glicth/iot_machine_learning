"""Tests for SignalCoherenceChecker domain service.

Covers the three main coherence scenarios:
- Anomaly True + Prediction Normal → Conflict detected
- Anomaly True + Prediction Anomalous → Coherent
- Anomaly False + Prediction Normal → Coherent
"""

from __future__ import annotations

import pytest

from iot_machine_learning.domain.services.signal_coherence_checker import (
    CoherenceResult,
    SignalCoherenceChecker,
)


class TestCoherenceCheckerBasics:
    """Basic construction and sanity checks."""

    def test_checker_construction(self):
        """SignalCoherenceChecker can be instantiated."""
        checker = SignalCoherenceChecker()
        assert checker is not None
        assert checker.CONFLICT_CONFIDENCE_PENALTY == 0.3

    def test_coherence_result_dataclass(self):
        """CoherenceResult is a proper frozen dataclass."""
        result = CoherenceResult(
            is_coherent=True,
            conflict_type=None,
            resolved_value=42.0,
            resolved_confidence=0.8,
            resolution_reason="Test reason",
        )
        assert result.is_coherent is True
        assert result.conflict_type is None
        assert result.resolved_value == 42.0
        assert result.resolved_confidence == 0.8


class TestNoAnomalyCoherent:
    """When no anomaly detected, always coherent."""

    def test_no_anomaly_normal_prediction(self):
        """No anomaly + normal prediction → coherent."""
        checker = SignalCoherenceChecker()
        result = checker.check(
            predicted_value=25.0,
            predicted_confidence=0.85,
            is_anomaly=False,
            anomaly_score=0.0,
            historical_values=[20.0, 22.0, 24.0, 26.0, 28.0],
        )
        assert result.is_coherent is True
        assert result.conflict_type is None
        assert result.resolved_confidence == 0.85
        assert "No anomaly detected" in result.resolution_reason

    def test_no_anomaly_without_historical(self):
        """No anomaly works even without historical values."""
        checker = SignalCoherenceChecker()
        result = checker.check(
            predicted_value=100.0,
            predicted_confidence=0.7,
            is_anomaly=False,
            anomaly_score=0.0,
            historical_values=None,
        )
        assert result.is_coherent is True
        assert result.conflict_type is None


class TestAnomalyPredictionConflict:
    """Conflict: Anomaly detected but prediction in normal range."""

    def test_anomaly_true_prediction_normal_conflict(self):
        """Anomaly=True + prediction in normal range → conflict detected."""
        checker = SignalCoherenceChecker()
        # Historical range: 20-30, prediction 25 is within normal range
        historical = [20.0, 22.0, 24.0, 26.0, 28.0, 30.0]
        result = checker.check(
            predicted_value=25.0,
            predicted_confidence=0.85,
            is_anomaly=True,
            anomaly_score=0.75,
            historical_values=historical,
        )
        assert result.is_coherent is False
        assert result.conflict_type == "anomaly_prediction_conflict"
        # Confidence should be penalized
        assert result.resolved_confidence <= 0.3
        assert result.resolved_confidence >= 0.1  # Floor
        assert "Anomaly detected" in result.resolution_reason
        assert "normal historical range" in result.resolution_reason

    def test_conflict_lowers_confidence(self):
        """Conflict always lowers confidence to penalty threshold."""
        checker = SignalCoherenceChecker()
        historical = [20.0, 22.0, 24.0, 26.0, 28.0]
        result = checker.check(
            predicted_value=25.0,
            predicted_confidence=0.95,  # High confidence should drop
            is_anomaly=True,
            anomaly_score=0.8,
            historical_values=historical,
        )
        assert result.resolved_confidence == 0.3  # Penalty cap

    def test_conflict_with_high_confidence_floor(self):
        """Conflict confidence has floor of 0.1."""
        checker = SignalCoherenceChecker()
        # Modify the penalty to be lower than floor
        checker.CONFLICT_CONFIDENCE_PENALTY = 0.05
        historical = [20.0, 22.0, 24.0, 26.0, 28.0]
        result = checker.check(
            predicted_value=25.0,
            predicted_confidence=0.1,
            is_anomaly=True,
            anomaly_score=0.8,
            historical_values=historical,
        )
        # Should use floor of 0.1, not 0.05
        assert result.resolved_confidence == 0.1


class TestAnomalyCoherent:
    """Anomaly detected and prediction consistent → coherent."""

    def test_anomaly_true_prediction_anomalous_coherent(self):
        """Anomaly=True + prediction outside normal range → coherent."""
        checker = SignalCoherenceChecker()
        # Historical range: 20-30 via IQR, prediction 100 is way outside
        historical = [20.0, 22.0, 24.0, 26.0, 28.0]
        result = checker.check(
            predicted_value=100.0,
            predicted_confidence=0.75,
            is_anomaly=True,
            anomaly_score=0.85,
            historical_values=historical,
        )
        assert result.is_coherent is True
        assert result.conflict_type is None
        assert result.resolved_confidence == 0.75  # Unchanged
        assert "consistent with anomalous regime" in result.resolution_reason

    def test_anomaly_with_extreme_value_coherent(self):
        """Extreme predicted value with anomaly is coherent."""
        checker = SignalCoherenceChecker()
        historical = [10.0, 12.0, 14.0, 16.0, 18.0]
        result = checker.check(
            predicted_value=500.0,
            predicted_confidence=0.6,
            is_anomaly=True,
            anomaly_score=0.9,
            historical_values=historical,
        )
        assert result.is_coherent is True
        assert result.resolved_value == 500.0


class TestNormalRangeCalculation:
    """Internal _is_in_normal_range logic."""

    def test_insufficient_data_assumes_normal(self):
        """With <4 historical values, assume in normal range (conservative)."""
        checker = SignalCoherenceChecker()
        result = checker.check(
            predicted_value=25.0,
            predicted_confidence=0.8,
            is_anomaly=True,
            anomaly_score=0.7,
            historical_values=[20.0, 30.0],  # Only 2 values
        )
        # With insufficient data, assumes normal range → conflict
        assert result.is_coherent is False
        assert result.conflict_type == "anomaly_prediction_conflict"

    def test_empty_historical_assumes_normal(self):
        """Empty historical values assumes in normal range."""
        checker = SignalCoherenceChecker()
        result = checker.check(
            predicted_value=25.0,
            predicted_confidence=0.8,
            is_anomaly=True,
            anomaly_score=0.7,
            historical_values=[],
        )
        assert result.is_coherent is False

    def test_none_historical_assumes_normal(self):
        """None historical values assumes in normal range."""
        checker = SignalCoherenceChecker()
        result = checker.check(
            predicted_value=25.0,
            predicted_confidence=0.8,
            is_anomaly=True,
            anomaly_score=0.7,
            historical_values=None,
        )
        assert result.is_coherent is False


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_boundary_iqr_edge(self):
        """Value exactly at IQR boundary."""
        checker = SignalCoherenceChecker()
        # Historical: Q1=22, Q3=26, IQR=4, bounds=[16, 32]
        historical = [20.0, 21.0, 22.0, 26.0, 27.0, 28.0]
        # Value at 32 is at boundary (32 = Q3 + 1.5*IQR)
        result = checker.check(
            predicted_value=32.0,
            predicted_confidence=0.8,
            is_anomaly=True,
            anomaly_score=0.7,
            historical_values=historical,
        )
        # At boundary is considered in normal range
        assert result.is_coherent is False
        assert result.conflict_type == "anomaly_prediction_conflict"

    def test_just_outside_iqr(self):
        """Value just outside IQR range."""
        checker = SignalCoherenceChecker()
        historical = [20.0, 21.0, 22.0, 26.0, 27.0, 28.0]
        # Value at 100 is clearly outside upper bound (32)
        result = checker.check(
            predicted_value=100.0,
            predicted_confidence=0.8,
            is_anomaly=True,
            anomaly_score=0.7,
            historical_values=historical,
        )
        # Outside IQR range → coherent with anomaly
        assert result.is_coherent is True
        assert result.conflict_type is None
