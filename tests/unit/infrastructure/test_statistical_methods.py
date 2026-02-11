"""Tests para statistical_methods.py.

Verifica funciones matemáticas PURAS — sin I/O, sin sklearn, sin estado.
"""

from __future__ import annotations

import pytest

from iot_machine_learning.infrastructure.ml.anomaly.statistical_methods import (
    TrainingStats,
    compute_consensus_confidence,
    compute_iqr_bounds,
    compute_iqr_vote,
    compute_training_stats,
    compute_z_score,
    compute_z_vote,
    weighted_vote,
)


# --- compute_training_stats ---

class TestComputeTrainingStats:
    """Tests para cálculo de estadísticas de entrenamiento."""

    def test_returns_training_stats(self) -> None:
        stats = compute_training_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert isinstance(stats, TrainingStats)

    def test_mean_correct(self) -> None:
        stats = compute_training_stats([10.0, 20.0, 30.0])
        assert abs(stats.mean - 20.0) < 0.01

    def test_std_correct(self) -> None:
        stats = compute_training_stats([10.0, 20.0, 30.0])
        # std = sqrt(((10-20)^2 + (20-20)^2 + (30-20)^2) / 3) = sqrt(200/3) ≈ 8.165
        assert abs(stats.std - 8.165) < 0.01

    def test_std_min_clamp(self) -> None:
        """Valores idénticos → std clamped a 1e-9."""
        stats = compute_training_stats([5.0, 5.0, 5.0])
        assert stats.std == pytest.approx(1e-9)

    def test_quartiles(self) -> None:
        values = list(range(100))
        stats = compute_training_stats(values)
        assert stats.q1 == 25.0
        assert stats.q3 == 75.0
        assert stats.iqr == 50.0

    def test_empty_values(self) -> None:
        stats = compute_training_stats([])
        assert stats.mean == 0.0
        assert stats.std == pytest.approx(1e-9)


# --- compute_z_score ---

class TestComputeZScore:
    """Tests para cálculo de Z-score."""

    def test_zero_deviation(self) -> None:
        assert compute_z_score(20.0, 20.0, 5.0) == 0.0

    def test_one_sigma(self) -> None:
        assert compute_z_score(25.0, 20.0, 5.0) == pytest.approx(1.0)

    def test_three_sigma(self) -> None:
        assert compute_z_score(35.0, 20.0, 5.0) == pytest.approx(3.0)

    def test_negative_deviation(self) -> None:
        """Z-score es absoluto."""
        assert compute_z_score(10.0, 20.0, 5.0) == pytest.approx(2.0)

    def test_tiny_std_returns_zero(self) -> None:
        assert compute_z_score(25.0, 20.0, 1e-15) == 0.0


# --- compute_z_vote ---

class TestComputeZVote:
    """Tests para conversión Z-score → voto."""

    def test_below_2_sigma(self) -> None:
        assert compute_z_vote(1.5) == 0.0

    def test_at_2_sigma(self) -> None:
        assert compute_z_vote(2.0) == 0.0

    def test_between_2_and_3_sigma(self) -> None:
        assert compute_z_vote(2.5) == pytest.approx(0.5)

    def test_at_3_sigma(self) -> None:
        assert compute_z_vote(3.0) == pytest.approx(1.0)

    def test_above_3_sigma(self) -> None:
        assert compute_z_vote(5.0) == 1.0


# --- compute_iqr_bounds ---

class TestComputeIqrBounds:
    """Tests para cálculo de límites IQR."""

    def test_standard_bounds(self) -> None:
        lower, upper = compute_iqr_bounds(25.0, 75.0, 50.0)
        assert lower == pytest.approx(-50.0)
        assert upper == pytest.approx(150.0)

    def test_zero_iqr(self) -> None:
        lower, upper = compute_iqr_bounds(50.0, 50.0, 0.0)
        assert lower == 50.0
        assert upper == 50.0


# --- compute_iqr_vote ---

class TestComputeIqrVote:
    """Tests para voto IQR."""

    def test_within_range(self) -> None:
        assert compute_iqr_vote(50.0, 25.0, 75.0, 50.0) == 0.0

    def test_below_range(self) -> None:
        assert compute_iqr_vote(-100.0, 25.0, 75.0, 50.0) == 1.0

    def test_above_range(self) -> None:
        assert compute_iqr_vote(200.0, 25.0, 75.0, 50.0) == 1.0

    def test_zero_iqr(self) -> None:
        assert compute_iqr_vote(100.0, 50.0, 50.0, 0.0) == 0.0


# --- weighted_vote ---

class TestWeightedVote:
    """Tests para voting ponderado."""

    def test_all_zeros(self) -> None:
        votes = {"a": 0.0, "b": 0.0}
        weights = {"a": 0.5, "b": 0.5}
        assert weighted_vote(votes, weights) == 0.0

    def test_all_ones(self) -> None:
        votes = {"a": 1.0, "b": 1.0}
        weights = {"a": 0.5, "b": 0.5}
        assert weighted_vote(votes, weights) == pytest.approx(1.0)

    def test_weighted_average(self) -> None:
        votes = {"a": 1.0, "b": 0.0}
        weights = {"a": 0.75, "b": 0.25}
        # (1.0 * 0.75 + 0.0 * 0.25) / (0.75 + 0.25) = 0.75
        assert weighted_vote(votes, weights) == pytest.approx(0.75)

    def test_missing_weight_uses_default(self) -> None:
        votes = {"a": 1.0, "unknown": 1.0}
        weights = {"a": 0.5}
        # (1.0 * 0.5 + 1.0 * 0.1) / (0.5 + 0.1) = 1.0
        assert weighted_vote(votes, weights, default_weight=0.1) == pytest.approx(1.0)

    def test_empty_votes(self) -> None:
        assert weighted_vote({}, {"a": 0.5}) == 0.0


# --- compute_consensus_confidence ---

class TestComputeConsensusConfidence:
    """Tests para confianza basada en consenso."""

    def test_perfect_consensus_all_zero(self) -> None:
        votes = {"a": 0.0, "b": 0.0, "c": 0.0}
        assert compute_consensus_confidence(votes) == pytest.approx(1.0)

    def test_perfect_consensus_all_one(self) -> None:
        votes = {"a": 1.0, "b": 1.0, "c": 1.0}
        assert compute_consensus_confidence(votes) == pytest.approx(1.0)

    def test_no_consensus(self) -> None:
        votes = {"a": 0.0, "b": 1.0}
        conf = compute_consensus_confidence(votes)
        assert 0.5 <= conf < 1.0

    def test_single_vote(self) -> None:
        assert compute_consensus_confidence({"a": 0.5}) == 0.6

    def test_minimum_confidence(self) -> None:
        """Confianza nunca baja de 0.5."""
        votes = {"a": 0.0, "b": 1.0}
        assert compute_consensus_confidence(votes) >= 0.5
