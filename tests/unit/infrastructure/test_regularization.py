"""Tests for L2 Regularization in BayesianWeightTracker."""

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.bayesian_weight_tracker.regularization import (
    apply_l2_regularization,
    compute_regularization_strength,
)


class TestApplyL2Regularization:
    """Test L2 regularization toward uniform weights."""
    
    def test_zero_regularization_returns_original(self):
        """λ=0 should return original accuracies."""
        accuracies = {"engine_a": 0.8, "engine_b": 0.6}
        engines = ["engine_a", "engine_b"]
        
        result = apply_l2_regularization(accuracies, engines, regularization_strength=0.0)
        
        assert result == accuracies
    
    def test_full_regularization_returns_uniform(self):
        """λ=1 should return uniform weights."""
        accuracies = {"engine_a": 0.9, "engine_b": 0.3}
        engines = ["engine_a", "engine_b"]
        
        result = apply_l2_regularization(accuracies, engines, regularization_strength=1.0)
        
        assert result["engine_a"] == pytest.approx(0.5, abs=1e-6)
        assert result["engine_b"] == pytest.approx(0.5, abs=1e-6)
    
    def test_partial_regularization_pulls_toward_uniform(self):
        """λ=0.2 should pull weights toward uniform."""
        accuracies = {"engine_a": 1.0, "engine_b": 0.0}
        engines = ["engine_a", "engine_b"]
        
        result = apply_l2_regularization(accuracies, engines, regularization_strength=0.2)
        
        # engine_a: (1-0.2)*1.0 + 0.2*0.5 = 0.8 + 0.1 = 0.9
        # engine_b: (1-0.2)*0.0 + 0.2*0.5 = 0.0 + 0.1 = 0.1
        assert result["engine_a"] == pytest.approx(0.9, abs=1e-6)
        assert result["engine_b"] == pytest.approx(0.1, abs=1e-6)
    
    def test_three_engines(self):
        """Regularization with 3 engines."""
        accuracies = {"e1": 0.9, "e2": 0.6, "e3": 0.3}
        engines = ["e1", "e2", "e3"]
        
        result = apply_l2_regularization(accuracies, engines, regularization_strength=0.1)
        
        # Uniform target: 1/3 ≈ 0.333
        # e1: 0.9*0.9 + 0.1*0.333 = 0.81 + 0.0333 = 0.8433
        assert result["e1"] == pytest.approx(0.8433, abs=1e-3)
    
    def test_empty_engines_returns_empty(self):
        """Empty engine list returns empty dict."""
        result = apply_l2_regularization({}, [], regularization_strength=0.5)
        assert result == {}
    
    def test_negative_lambda_treated_as_zero(self):
        """Negative λ should be clamped to 0."""
        accuracies = {"e1": 0.8}
        engines = ["e1"]
        
        result = apply_l2_regularization(accuracies, engines, regularization_strength=-0.5)
        
        assert result == accuracies
    
    def test_lambda_above_one_clamped(self):
        """λ > 1 should be clamped to 1."""
        accuracies = {"e1": 0.9, "e2": 0.1}
        engines = ["e1", "e2"]
        
        result = apply_l2_regularization(accuracies, engines, regularization_strength=1.5)
        
        # Should behave like λ=1 (uniform)
        assert result["e1"] == pytest.approx(0.5, abs=1e-6)
        assert result["e2"] == pytest.approx(0.5, abs=1e-6)


class TestComputeRegularizationStrength:
    """Test drift-aware adaptive regularization strength."""

    def test_no_drift_equals_base(self):
        """With drift_score=0, strength equals base_strength."""
        strength = compute_regularization_strength(n_updates=0, base_strength=0.05)
        assert strength == pytest.approx(0.05, abs=1e-6)

    def test_strength_increases_with_drift(self):
        """Strength should increase monotonically with drift_score."""
        s0 = compute_regularization_strength(0, base_strength=0.05, drift_score=0.0)
        s1 = compute_regularization_strength(0, base_strength=0.05, drift_score=1.0)
        s2 = compute_regularization_strength(0, base_strength=0.05, drift_score=2.0)

        assert s1 > s0
        assert s2 > s1
        assert s1 == pytest.approx(0.10, abs=1e-6)  # 0.05 * (1 + 1.0)
        assert s2 == pytest.approx(0.15, abs=1e-6)  # 0.05 * (1 + 2.0)

    def test_strength_never_below_min(self):
        """Strength should not go below min_strength even with zero drift."""
        strength = compute_regularization_strength(
            n_updates=0,
            base_strength=0.0005,
            min_strength=0.001,
        )
        assert strength >= 0.001

    def test_drift_rate_parameter_backward_compat(self):
        """drift_rate parameter is retained for backward-compat signature."""
        # drift_rate is accepted but unused in new formula
        s1 = compute_regularization_strength(100, base_strength=0.05, decay_rate=0.5)
        s2 = compute_regularization_strength(100, base_strength=0.05, decay_rate=0.99)
        # Without drift, both should equal base_strength
        assert s1 == pytest.approx(0.05, abs=1e-6)
        assert s2 == pytest.approx(0.05, abs=1e-6)

    def test_negative_drift_treated_as_zero(self):
        """Negative drift_score should be treated as zero (abs value)."""
        s_neg = compute_regularization_strength(0, base_strength=0.05, drift_score=-1.0)
        s_pos = compute_regularization_strength(0, base_strength=0.05, drift_score=1.0)
        assert s_neg == s_pos
