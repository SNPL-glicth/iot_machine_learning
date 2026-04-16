"""Tests para GatingNetwork y estrategias.

Cobertura: RegimeBasedGating, GatingProbs, ContextVector.
"""

import pytest
import math

from ..gating.base import GatingProbs, ContextVector
from ..gating.regime_based import RegimeBasedGating, RegimeRoutingRule


class TestGatingProbs:
    """Tests para dataclass GatingProbs."""
    
    def test_valid_probabilities(self):
        probs = GatingProbs(
            probabilities={"e1": 0.7, "e2": 0.3},
            entropy=0.6,
            top_expert="e1"
        )
        
        assert probs.max_probability == 0.7
        assert probs.min_probability == 0.3
    
    def test_invalid_probabilities_sum(self):
        with pytest.raises(ValueError) as exc_info:
            GatingProbs(
                probabilities={"e1": 0.5, "e2": 0.3},  # Suma 0.8 != 1.0
                entropy=0.5,
                top_expert="e1"
            )
        
        assert "deben sumar ~1.0" in str(exc_info.value)
    
    def test_get_top_k(self):
        probs = GatingProbs(
            probabilities={"e1": 0.6, "e2": 0.3, "e3": 0.1},
            entropy=0.9,
            top_expert="e1"
        )
        
        top2 = probs.get_top_k(2)
        
        assert top2 == ["e1", "e2"]
    
    def test_get_probability(self):
        probs = GatingProbs(
            probabilities={"e1": 0.7, "e2": 0.3},
            entropy=0.6,
            top_expert="e1"
        )
        
        assert probs.get_probability("e1") == 0.7
        assert probs.get_probability("nonexistent") == 0.0


class TestContextVector:
    """Tests para dataclass ContextVector."""
    
    def test_basic_creation(self):
        ctx = ContextVector(
            regime="stable",
            domain="iot",
            n_points=10,
            signal_features={"mean": 20.0, "std": 1.5}
        )
        
        assert ctx.regime == "stable"
        assert ctx.n_points == 10
    
    def test_to_array(self):
        ctx = ContextVector(
            regime="stable",
            domain="iot",
            n_points=10,
            signal_features={
                "mean": 20.0,
                "std": 1.5,
                "slope": 0.5,
                "curvature": 0.1,
                "noise_ratio": 0.05,
                "stability": 0.8,
            }
        )
        
        arr = ctx.to_array()
        
        assert len(arr) == 11  # 7 features + 4 one-hot regimes
        assert arr[0] == 10  # n_points
        assert arr[1] == 20.0  # mean
    
    def test_to_dict(self):
        ctx = ContextVector(
            regime="stable",
            domain="iot",
            n_points=10,
            signal_features={"mean": 20.0}
        )
        
        d = ctx.to_dict()
        
        assert d["regime"] == "stable"
        assert d["n_points"] == 10


class TestRegimeBasedGating:
    """Tests para RegimeBasedGating (heurístico)."""
    
    @pytest.fixture
    def gating(self):
        return RegimeBasedGating.with_default_rules(["baseline", "statistical", "taylor"])
    
    def test_route_stable(self, gating):
        ctx = ContextVector(
            regime="stable",
            domain="iot",
            n_points=10,
            signal_features={}
        )
        
        result = gating.route(ctx)
        
        assert result.top_expert == "baseline"
        assert result.probabilities["baseline"] > 0.5
        assert "Régimen estable" in result.metadata["rationale"]
    
    def test_route_trending(self, gating):
        ctx = ContextVector(
            regime="trending",
            domain="iot",
            n_points=10,
            signal_features={}
        )
        
        result = gating.route(ctx)
        
        assert result.top_expert == "statistical"
        assert result.probabilities["statistical"] > 0.5
    
    def test_route_volatile(self, gating):
        ctx = ContextVector(
            regime="volatile",
            domain="iot",
            n_points=10,
            signal_features={}
        )
        
        result = gating.route(ctx)
        
        assert result.top_expert == "taylor"
        assert result.probabilities["taylor"] >= 0.7
    
    def test_route_unknown_regime_uses_default(self, gating):
        ctx = ContextVector(
            regime="unknown_regime",
            domain="iot",
            n_points=10,
            signal_features={}
        )
        
        result = gating.route(ctx)
        
        # Usa default rule
        assert result.metadata["rule_used"] == "unknown"
    
    def test_probabilities_sum_to_one(self, gating):
        ctx = ContextVector(
            regime="stable",
            domain="iot",
            n_points=10,
            signal_features={}
        )
        
        result = gating.route(ctx)
        
        total = sum(result.probabilities.values())
        assert 0.99 <= total <= 1.01
    
    def test_entropy_calculation(self, gating):
        ctx = ContextVector(
            regime="volatile",
            domain="iot",
            n_points=10,
            signal_features={}
        )
        
        result = gating.route(ctx)
        
        # Volatile tiene alta confianza en taylor → baja entropía
        assert result.entropy < 1.0
    
    def test_explain(self, gating):
        ctx = ContextVector(
            regime="stable",
            domain="iot",
            n_points=10,
            signal_features={}
        )
        
        result = gating.route(ctx)
        explanation = gating.explain(ctx, result)
        
        assert "stable" in explanation.lower()
        assert "baseline" in explanation
        assert "Entropía" in explanation


class TestRegimeRoutingRule:
    """Tests para RegimeRoutingRule."""
    
    def test_valid_rule(self):
        rule = RegimeRoutingRule(
            regime="test",
            expert_weights={"e1": 0.7, "e2": 0.3},
            rationale="Test rule"
        )
        
        assert rule.regime == "test"
        assert rule.rationale == "Test rule"
    
    def test_invalid_weights_zero(self):
        with pytest.raises(ValueError):
            RegimeRoutingRule(
                regime="test",
                expert_weights={},
                rationale="Empty"
            )
    
    def test_to_probabilities(self):
        rule = RegimeRoutingRule(
            regime="test",
            expert_weights={"e1": 7.0, "e2": 3.0},
            rationale="Test"
        )
        
        all_experts = ["e1", "e2", "e3"]
        probs = rule.to_probabilities(all_experts)
        
        assert probs["e1"] == 0.7
        assert probs["e2"] == 0.3
        assert probs["e3"] == 0.0


class TestRegimeBasedGatingCustomRules:
    """Tests con reglas personalizadas."""
    
    def test_add_custom_rule(self):
        gating = RegimeBasedGating.with_default_rules(["baseline"])
        
        custom_rule = RegimeRoutingRule(
            regime="custom",
            expert_weights={"baseline": 1.0},
            rationale="Custom rule"
        )
        
        gating.add_rule(custom_rule)
        
        assert "custom" in gating.list_rules()
    
    def test_remove_rule(self):
        gating = RegimeBasedGating.with_default_rules(["baseline"])
        
        removed = gating.remove_rule("stable")
        
        assert removed is True
        assert "stable" not in gating.list_rules()
    
    def test_remove_nonexistent_rule(self):
        gating = RegimeBasedGating.with_default_rules(["baseline"])
        
        removed = gating.remove_rule("nonexistent")
        
        assert removed is False
    
    def test_conservative_factory(self):
        gating = RegimeBasedGating.conservative(["baseline", "statistical", "taylor"])
        
        ctx = ContextVector(regime="stable", domain="iot", n_points=10, signal_features={})
        result = gating.route(ctx)
        
        # Configuración conservadora: solo baseline para stable
        assert result.probabilities["baseline"] == 1.0


class TestGatingStats:
    """Tests de estadísticas del gating."""
    
    def test_initial_stats(self):
        gating = RegimeBasedGating.with_default_rules(["e1", "e2"])
        
        stats = gating.get_stats()
        
        assert stats["routing_count"] == 0
        assert stats["expert_count"] == 2
        assert stats["gating_type"] == "RegimeBasedGating"
    
    def test_stats_after_routing(self):
        # Crear gating con reglas que solo referencian e1 y e2
        gating = RegimeBasedGating(
            expert_ids=["e1", "e2"],
            rules={
                "stable": RegimeRoutingRule(
                    regime="stable",
                    expert_weights={"e1": 0.7, "e2": 0.3},
                    rationale="Test rule"
                )
            }
        )
        
        ctx = ContextVector(regime="stable", domain="iot", n_points=10, signal_features={})
        gating.route(ctx)
        gating.route(ctx)
        
        stats = gating.get_stats()
        
        assert stats["routing_count"] == 2


class TestGatingErrors:
    """Tests de manejo de errores."""
    
    def test_empty_expert_ids(self):
        gating = RegimeBasedGating(expert_ids=[])
        
        ctx = ContextVector(regime="stable", domain="iot", n_points=10, signal_features={})
        
        with pytest.raises(ValueError) as exc_info:
            gating.route(ctx)
        
        assert "Expert IDs no configurados" in str(exc_info.value)
    
    def test_missing_regime(self):
        gating = RegimeBasedGating.with_default_rules(["e1"])
        
        ctx = ContextVector(regime="", domain="iot", n_points=10, signal_features={})
        
        with pytest.raises(ValueError) as exc_info:
            gating.route(ctx)
        
        assert "debe tener regime definido" in str(exc_info.value)
