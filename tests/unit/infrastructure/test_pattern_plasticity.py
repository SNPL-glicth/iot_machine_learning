"""Tests for PatternPlasticityTracker."""

from __future__ import annotations

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.universal.analysis.pattern_plasticity import (
    PatternPlasticityTracker,
    PATTERN_DELTA_SPIKES,
    PATTERN_CHANGE_POINTS,
    PATTERN_REGIME_CHANGES,
    ALL_PATTERNS,
)


class TestPatternPlasticityBasic:
    """Basic functionality tests."""

    def test_tracker_instantiates(self) -> None:
        """Tracker should instantiate without errors."""
        tracker = PatternPlasticityTracker()
        assert tracker is not None

    def test_initial_weights_uniform(self) -> None:
        """Initial weights should be uniform for new domain."""
        tracker = PatternPlasticityTracker()
        
        weights = tracker.get_pattern_weights("infrastructure")
        
        assert len(weights) == 3
        assert all(p in weights for p in ALL_PATTERNS)
        # Should be ~0.333 each
        assert all(0.3 < w < 0.35 for w in weights.values())

    def test_record_outcome_updates_weight(self) -> None:
        """Recording positive outcome should increase pattern weight."""
        tracker = PatternPlasticityTracker(alpha=0.5)  # High alpha for faster learning
        
        # Record delta_spikes as predictive multiple times
        for _ in range(5):
            tracker.record_pattern_outcome(
                domain="infrastructure",
                pattern_name=PATTERN_DELTA_SPIKES,
                was_predictive=True,
            )
        
        weights = tracker.get_pattern_weights("infrastructure")
        
        # delta_spikes should have higher weight than others
        assert weights[PATTERN_DELTA_SPIKES] > weights[PATTERN_CHANGE_POINTS]
        assert weights[PATTERN_DELTA_SPIKES] > weights[PATTERN_REGIME_CHANGES]

    def test_record_negative_outcome_decreases_weight(self) -> None:
        """Recording negative outcome should decrease pattern weight."""
        tracker = PatternPlasticityTracker(alpha=0.5)
        
        # Record change_points as NOT predictive
        for _ in range(5):
            tracker.record_pattern_outcome(
                domain="security",
                pattern_name=PATTERN_CHANGE_POINTS,
                was_predictive=False,
            )
        
        weights = tracker.get_pattern_weights("security")
        
        # change_points should have lower weight than others
        assert weights[PATTERN_CHANGE_POINTS] < weights[PATTERN_DELTA_SPIKES]
        assert weights[PATTERN_CHANGE_POINTS] < weights[PATTERN_REGIME_CHANGES]


class TestPatternPlasticityPerDomain:
    """Test domain-specific learning."""

    def test_different_domains_independent(self) -> None:
        """Different domains should learn independently."""
        tracker = PatternPlasticityTracker(alpha=0.5)
        
        # Infrastructure: delta_spikes predictive
        for _ in range(5):
            tracker.record_pattern_outcome(
                domain="infrastructure",
                pattern_name=PATTERN_DELTA_SPIKES,
                was_predictive=True,
            )
        
        # Security: change_points predictive
        for _ in range(5):
            tracker.record_pattern_outcome(
                domain="security",
                pattern_name=PATTERN_CHANGE_POINTS,
                was_predictive=True,
            )
        
        infra_weights = tracker.get_pattern_weights("infrastructure")
        sec_weights = tracker.get_pattern_weights("security")
        
        # Each domain should have different pattern preferences
        assert infra_weights[PATTERN_DELTA_SPIKES] > infra_weights[PATTERN_CHANGE_POINTS]
        assert sec_weights[PATTERN_CHANGE_POINTS] > sec_weights[PATTERN_DELTA_SPIKES]

    def test_new_domain_starts_uniform(self) -> None:
        """New domain should start with uniform weights."""
        tracker = PatternPlasticityTracker()
        
        # Train infrastructure
        tracker.record_pattern_outcome(
            domain="infrastructure",
            pattern_name=PATTERN_DELTA_SPIKES,
            was_predictive=True,
        )
        
        # Query new domain
        trading_weights = tracker.get_pattern_weights("trading")
        
        # Should be uniform (no history)
        assert all(0.3 < w < 0.35 for w in trading_weights.values())


class TestPatternPlasticityWeightNormalization:
    """Test weight normalization."""

    def test_weights_sum_to_one(self) -> None:
        """Pattern weights should always sum to 1.0."""
        tracker = PatternPlasticityTracker()
        
        # Record various outcomes
        tracker.record_pattern_outcome("test", PATTERN_DELTA_SPIKES, True)
        tracker.record_pattern_outcome("test", PATTERN_CHANGE_POINTS, False)
        tracker.record_pattern_outcome("test", PATTERN_REGIME_CHANGES, True)
        
        weights = tracker.get_pattern_weights("test")
        
        total = sum(weights.values())
        assert 0.99 < total < 1.01  # Allow small floating point error

    def test_minimum_weight_floor_enforced(self) -> None:
        """No weight should fall below minimum floor."""
        tracker = PatternPlasticityTracker(min_weight=0.1, alpha=0.8)
        
        # Try to suppress change_points
        for _ in range(20):
            tracker.record_pattern_outcome(
                domain="test",
                pattern_name=PATTERN_CHANGE_POINTS,
                was_predictive=False,
            )
        
        weights = tracker.get_pattern_weights("test")
        
        # Even after many negative outcomes, should respect floor
        assert weights[PATTERN_CHANGE_POINTS] >= 0.1


class TestPatternPlasticityGracefulFail:
    """Test error handling."""

    def test_unknown_pattern_handled(self) -> None:
        """Unknown pattern name should be handled gracefully."""
        tracker = PatternPlasticityTracker()
        
        # Should not crash
        tracker.record_pattern_outcome(
            domain="test",
            pattern_name="unknown_pattern",
            was_predictive=True,
        )
        
        # Should still return uniform weights
        weights = tracker.get_pattern_weights("test")
        assert len(weights) == 3

    def test_empty_domain_handled(self) -> None:
        """Empty domain should return uniform weights."""
        tracker = PatternPlasticityTracker()
        
        weights = tracker.get_pattern_weights("")
        
        assert len(weights) == 3
        assert all(0.3 < w < 0.35 for w in weights.values())


class TestPatternPlasticityReset:
    """Test reset functionality."""

    def test_reset_specific_domain(self) -> None:
        """Reset should clear specific domain only."""
        tracker = PatternPlasticityTracker()
        
        # Train two domains
        tracker.record_pattern_outcome("infra", PATTERN_DELTA_SPIKES, True)
        tracker.record_pattern_outcome("security", PATTERN_CHANGE_POINTS, True)
        
        # Reset only infrastructure
        tracker.reset(domain="infra")
        
        # Infrastructure should have no history
        assert not tracker.has_history("infra")
        # Security should still have history
        assert tracker.has_history("security")

    def test_reset_all_domains(self) -> None:
        """Reset without domain should clear all."""
        tracker = PatternPlasticityTracker()
        
        # Train multiple domains
        tracker.record_pattern_outcome("infra", PATTERN_DELTA_SPIKES, True)
        tracker.record_pattern_outcome("security", PATTERN_CHANGE_POINTS, True)
        tracker.record_pattern_outcome("trading", PATTERN_REGIME_CHANGES, True)
        
        # Reset all
        tracker.reset()
        
        # All should have no history
        assert not tracker.has_history("infra")
        assert not tracker.has_history("security")
        assert not tracker.has_history("trading")


class TestPatternPlasticityStats:
    """Test statistics and monitoring."""

    def test_get_stats_for_domain(self) -> None:
        """get_stats should return domain statistics."""
        tracker = PatternPlasticityTracker()
        
        tracker.record_pattern_outcome("test", PATTERN_DELTA_SPIKES, True)
        
        stats = tracker.get_stats(domain="test")
        
        assert stats["domain"] == "test"
        assert stats["has_history"] is True
        assert "weights" in stats

    def test_get_global_stats(self) -> None:
        """get_stats without domain should return global stats."""
        tracker = PatternPlasticityTracker(max_domains=10)
        
        tracker.record_pattern_outcome("infra", PATTERN_DELTA_SPIKES, True)
        tracker.record_pattern_outcome("security", PATTERN_CHANGE_POINTS, True)
        
        stats = tracker.get_stats()
        
        assert stats["total_domains"] == 2
        assert stats["max_domains"] == 10
        assert "infra" in stats["domains"]
        assert "security" in stats["domains"]
