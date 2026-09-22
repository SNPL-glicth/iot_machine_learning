"""Tests para verificar pesos deterministas de RegimeBasedGating.

Verifica:
- 3 regímenes: stable, trending, volatile
- Suma de probabilidades = 1.0
- Pesos exactos según especificación ZENIN MoE
"""

import pytest
import math

from domain.model.context_vector import ContextVector
from infrastructure.ml.moe.gating.regime_based import RegimeBasedGating, RegimeRoutingRule
from infrastructure.ml.moe.gating.base import GatingProbs


class TestRegimeWeightsSpecification:
    """Test weights according to ZENIN MoE specification."""
    
    @pytest.fixture
    def gating(self):
        """Gating with default rules for 3 experts."""
        return RegimeBasedGating.with_default_rules(
            ["baseline", "statistical", "taylor"]
        )
    
    def test_stable_weights(self, gating):
        """Regime 'stable': baseline 0.85, statistical 0.10, taylor 0.05."""
        ctx = ContextVector(
            regime="stable",
            domain="iot",
            n_points=10,
            signal_features={}
        )
        
        probs = gating.route(ctx)
        
        # Verify exact weights
        assert probs.probabilities["baseline"] == 0.85
        assert probs.probabilities["statistical"] == 0.10
        assert probs.probabilities["taylor"] == 0.05
        
        # Top expert is baseline
        assert probs.top_expert == "baseline"
    
    def test_trending_weights(self, gating):
        """Regime 'trending': statistical 0.60, taylor 0.30, baseline 0.10."""
        ctx = ContextVector(
            regime="trending",
            domain="iot",
            n_points=10,
            signal_features={}
        )
        
        probs = gating.route(ctx)
        
        # Verify exact weights
        assert probs.probabilities["statistical"] == 0.60
        assert probs.probabilities["taylor"] == 0.30
        assert probs.probabilities["baseline"] == 0.10
        
        # Top expert is statistical
        assert probs.top_expert == "statistical"
    
    def test_volatile_weights(self, gating):
        """Regime 'volatile': taylor 0.70, statistical 0.20, baseline 0.10."""
        ctx = ContextVector(
            regime="volatile",
            domain="iot",
            n_points=10,
            signal_features={}
        )
        
        probs = gating.route(ctx)
        
        # Verify exact weights
        assert probs.probabilities["taylor"] == 0.70
        assert probs.probabilities["statistical"] == 0.20
        assert probs.probabilities["baseline"] == 0.10
        
        # Top expert is taylor
        assert probs.top_expert == "taylor"


class TestProbabilitiesSumToOne:
    """Verify all probability distributions sum to 1.0."""
    
    @pytest.fixture
    def gating(self):
        return RegimeBasedGating.with_default_rules(
            ["baseline", "statistical", "taylor"]
        )
    
    def test_stable_sums_to_one(self, gating):
        """Stable regime probabilities sum to 1.0."""
        ctx = ContextVector(regime="stable", domain="iot", n_points=10, signal_features={})
        probs = gating.route(ctx)
        
        total = sum(probs.probabilities.values())
        assert math.isclose(total, 1.0, abs_tol=1e-9)
    
    def test_trending_sums_to_one(self, gating):
        """Trending regime probabilities sum to 1.0."""
        ctx = ContextVector(regime="trending", domain="iot", n_points=10, signal_features={})
        probs = gating.route(ctx)
        
        total = sum(probs.probabilities.values())
        assert math.isclose(total, 1.0, abs_tol=1e-9)
    
    def test_volatile_sums_to_one(self, gating):
        """Volatile regime probabilities sum to 1.0."""
        ctx = ContextVector(regime="volatile", domain="iot", n_points=10, signal_features={})
        probs = gating.route(ctx)
        
        total = sum(probs.probabilities.values())
        assert math.isclose(total, 1.0, abs_tol=1e-9)


class TestDeterministicBehavior:
    """Verify deterministic routing (no randomness, no ML)."""
    
    def test_same_input_same_output(self):
        """Same context always produces same probabilities."""
        gating = RegimeBasedGating.with_default_rules(
            ["baseline", "statistical", "taylor"]
        )
        ctx = ContextVector(regime="volatile", domain="iot", n_points=10, signal_features={})
        
        # Run multiple times
        results = [gating.route(ctx) for _ in range(5)]
        
        # All results identical
        first = results[0]
        for result in results[1:]:
            assert result.probabilities == first.probabilities
            assert result.top_expert == first.top_expert
    
    def test_no_torch_no_ml(self):
        """Pure dictionary rules, no ML framework needed."""
        # Verify no torch import
        import sys
        assert "torch" not in sys.modules
        
        # Gating works without any ML library
        gating = RegimeBasedGating.with_default_rules(["baseline"])
        ctx = ContextVector(regime="stable", domain="iot", n_points=5, signal_features={})
        probs = gating.route(ctx)
        
        assert probs is not None
        assert "torch" not in sys.modules  # Still no torch


class TestRegimeRoutingRuleValidation:
    """Test RegimeRoutingRule validation."""
    
    def test_weights_must_sum_positive(self):
        """Rule requires positive weights."""
        with pytest.raises(ValueError, match="Pesos deben ser positivos"):
            RegimeRoutingRule(
                regime="test",
                expert_weights={},
                rationale="Empty weights"
            )
    
    def test_to_probabilities_normalization(self):
        """to_probabilities normalizes weights to sum 1.0."""
        rule = RegimeRoutingRule(
            regime="test",
            expert_weights={
                "expert_a": 2.0,
                "expert_b": 3.0,
            },
            rationale="Test"
        )
        
        probs = rule.to_probabilities(["expert_a", "expert_b", "expert_c"])
        
        # Weights 2:3 = 0.4:0.6
        assert probs["expert_a"] == 0.4
        assert probs["expert_b"] == 0.6
        assert probs["expert_c"] == 0.0  # Not in rule
        
        # Sum is 1.0
        assert math.isclose(sum(probs.values()), 1.0)


class TestFiveExpertRegimeWeights:
    """Verify harmonized routing when all 5 experts (including Kalman and Rosa Roja) are active."""

    @pytest.fixture
    def gating_five(self):
        return RegimeBasedGating.with_default_rules(
            ["baseline", "statistical", "taylor", "kalman", "rosa_roja"]
        )

    def test_five_experts_stable(self, gating_five):
        ctx = ContextVector(regime="stable", domain="iot", n_points=10, signal_features={})
        probs = gating_five.route(ctx)
        # All 5 have positive weights
        for e in ["baseline", "statistical", "taylor", "kalman", "rosa_roja"]:
            assert probs.probabilities[e] > 0.0
        assert probs.top_expert == "baseline"
        assert math.isclose(sum(probs.probabilities.values()), 1.0, abs_tol=1e-9)

    def test_five_experts_trending(self, gating_five):
        ctx = ContextVector(regime="trending", domain="iot", n_points=10, signal_features={})
        probs = gating_five.route(ctx)
        for e in ["baseline", "statistical", "taylor", "kalman", "rosa_roja"]:
            assert probs.probabilities[e] > 0.0
        assert probs.top_expert == "statistical"
        assert math.isclose(sum(probs.probabilities.values()), 1.0, abs_tol=1e-9)

    def test_five_experts_volatile(self, gating_five):
        ctx = ContextVector(regime="volatile", domain="iot", n_points=10, signal_features={})
        probs = gating_five.route(ctx)
        for e in ["baseline", "statistical", "taylor", "kalman", "rosa_roja"]:
            assert probs.probabilities[e] > 0.0
        assert probs.top_expert == "taylor"
        assert probs.probabilities["rosa_roja"] > probs.probabilities["statistical"]
        assert math.isclose(sum(probs.probabilities.values()), 1.0, abs_tol=1e-9)
