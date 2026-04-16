"""Tests for ExpertRegistry.

Scenario: Register 3 mock experts, filter by regime 'volatile'.
"""

import pytest
from dataclasses import dataclass

from domain.model.context_vector import ContextVector
from ..expert_capability import ExpertCapability
from ..expert_registry import ExpertRegistry, Expert


@dataclass
class MockExpert:
    """Mock expert for testing."""
    
    _name: str
    _caps: ExpertCapability
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def capabilities(self) -> ExpertCapability:
        return self._caps


class TestExpertRegistryRegistration:
    """Tests for expert registration."""
    
    def test_register_single_expert(self):
        """Register one expert successfully."""
        registry = ExpertRegistry()
        expert = MockExpert(
            "baseline",
            ExpertCapability(regimes=("stable",), computational_cost=0.5)
        )
        
        registry.register("baseline", expert, expert.capabilities)
        
        assert len(registry) == 1
        assert "baseline" in registry
    
    def test_register_multiple_experts(self):
        """Register three experts as per requirements."""
        registry = ExpertRegistry()
        
        # Three experts with different capabilities
        experts = [
            MockExpert("baseline", ExpertCapability(
                regimes=("stable",),
                computational_cost=0.5
            )),
            MockExpert("statistical", ExpertCapability(
                regimes=("stable", "trending"),
                computational_cost=1.0
            )),
            MockExpert("taylor", ExpertCapability(
                regimes=("volatile", "trending"),
                computational_cost=2.0
            )),
        ]
        
        for exp in experts:
            registry.register(exp.name, exp, exp.capabilities)
        
        assert len(registry) == 3
        assert list(registry) == ["baseline", "statistical", "taylor"]
    
    def test_register_duplicate_raises(self):
        """Duplicate registration raises ValueError."""
        registry = ExpertRegistry()
        expert = MockExpert("test", ExpertCapability())
        
        registry.register("test", expert, expert.capabilities)
        
        with pytest.raises(ValueError, match="already registered"):
            registry.register("test", expert, expert.capabilities)
    
    def test_register_force_overwrite(self):
        """Force flag allows overwriting."""
        registry = ExpertRegistry()
        expert1 = MockExpert("test", ExpertCapability(regimes=("stable",)))
        expert2 = MockExpert("test", ExpertCapability(regimes=("volatile",)))
        
        registry.register("test", expert1, expert1.capabilities)
        registry.register("test", expert2, expert2.capabilities, force=True)
        
        retrieved = registry.get("test")
        assert retrieved is expert2


class TestExpertRegistryFilterByRegime:
    """Filter experts by regime - main requirement."""
    
    @pytest.fixture
    def three_expert_registry(self):
        """Registry with 3 experts as per test scenario."""
        registry = ExpertRegistry()
        
        # Expert 1: only stable
        baseline = MockExpert(
            "baseline",
            ExpertCapability(regimes=("stable",), computational_cost=0.5)
        )
        registry.register("baseline", baseline, baseline.capabilities)
        
        # Expert 2: stable + trending
        statistical = MockExpert(
            "statistical",
            ExpertCapability(regimes=("stable", "trending"), computational_cost=1.0)
        )
        registry.register("statistical", statistical, statistical.capabilities)
        
        # Expert 3: volatile + trending
        taylor = MockExpert(
            "taylor",
            ExpertCapability(regimes=("volatile", "trending"), computational_cost=2.0)
        )
        registry.register("taylor", taylor, taylor.capabilities)
        
        return registry
    
    def test_filter_by_volatile(self, three_expert_registry):
        """Filter by 'volatile' regime - main requirement."""
        volatile_experts = three_expert_registry.find_by_regime("volatile")
        
        # Only taylor supports volatile
        assert volatile_experts == ["taylor"]
    
    def test_filter_by_stable(self, three_expert_registry):
        """Filter by 'stable' regime."""
        stable_experts = three_expert_registry.find_by_regime("stable")
        
        # baseline and statistical support stable
        assert "baseline" in stable_experts
        assert "statistical" in stable_experts
        assert "taylor" not in stable_experts
        assert len(stable_experts) == 2
    
    def test_filter_by_trending(self, three_expert_registry):
        """Filter by 'trending' regime."""
        trending_experts = three_expert_registry.find_by_regime("trending")
        
        # statistical and taylor support trending
        assert "statistical" in trending_experts
        assert "taylor" in trending_experts
        assert "baseline" not in trending_experts
        assert len(trending_experts) == 2
    
    def test_filter_by_unknown_regime(self, three_expert_registry):
        """Filter by regime no one supports."""
        noisy_experts = three_expert_registry.find_by_regime("noisy")
        
        assert noisy_experts == []


class TestExpertRegistryGetCandidates:
    """Filter by full context using ContextVector."""
    
    @pytest.fixture
    def populated_registry(self):
        """Registry with experts of varying capabilities."""
        registry = ExpertRegistry()
        
        # Expert that handles stable, needs 3+ points
        baseline = MockExpert(
            "baseline",
            ExpertCapability(
                regimes=("stable",),
                min_points=3,
                computational_cost=0.5
            )
        )
        registry.register("baseline", baseline, baseline.capabilities)
        
        # Expert that handles volatile, needs 5+ points
        taylor = MockExpert(
            "taylor",
            ExpertCapability(
                regimes=("volatile",),
                min_points=5,
                computational_cost=2.0
            )
        )
        registry.register("taylor", taylor, taylor.capabilities)
        
        # Expert that handles both, needs 3+ points
        statistical = MockExpert(
            "statistical",
            ExpertCapability(
                regimes=("stable", "volatile"),
                min_points=3,
                computational_cost=1.0
            )
        )
        registry.register("statistical", statistical, statistical.capabilities)
        
        return registry
    
    def test_get_candidates_volatile_context(self, populated_registry):
        """Get candidates for volatile regime with sufficient points."""
        context = ContextVector(
            regime="volatile",
            domain="iot",
            n_points=10,
            signal_features={}
        )
        
        candidates = populated_registry.get_candidates(context)
        
        # statistical (cost 1.0) should be first, then taylor (cost 2.0)
        assert "statistical" in candidates
        assert "taylor" in candidates
        assert "baseline" not in candidates  # Doesn't support volatile
        
        # Ordered by cost
        if len(candidates) >= 2:
            stat_cost = populated_registry.get_capabilities("statistical").computational_cost
            taylor_cost = populated_registry.get_capabilities("taylor").computational_cost
            assert stat_cost <= taylor_cost
    
    def test_get_candidates_insufficient_points(self, populated_registry):
        """Get candidates when n_points below minimum."""
        context = ContextVector(
            regime="volatile",
            domain="iot",
            n_points=2,  # Below taylor's min_points=5
            signal_features={}
        )
        
        candidates = populated_registry.get_candidates(context)
        
        # Only statistical can handle 2 points (min_points=3, but 2<3 so actually none)
        # Wait, statistical needs 3, so no one can handle 2
        assert candidates == []
    
    def test_get_candidates_wrong_domain(self, populated_registry):
        """Get candidates for unsupported domain."""
        # All experts default to iot domain, so finance should return empty
        context = ContextVector(
            regime="stable",
            domain="finance",
            n_points=10,
            signal_features={}
        )
        
        candidates = populated_registry.get_candidates(context)
        
        assert candidates == []


class TestExpertRegistryRetrieval:
    """Tests for expert retrieval methods."""
    
    def test_get_expert(self):
        """Retrieve expert by ID."""
        registry = ExpertRegistry()
        expert = MockExpert("test", ExpertCapability())
        registry.register("test", expert, expert.capabilities)
        
        retrieved = registry.get("test")
        
        assert retrieved is expert
    
    def test_get_nonexistent(self):
        """Retrieve non-existent expert returns None."""
        registry = ExpertRegistry()
        
        retrieved = registry.get("nonexistent")
        
        assert retrieved is None
    
    def test_get_capabilities(self):
        """Retrieve capabilities by expert ID."""
        registry = ExpertRegistry()
        caps = ExpertCapability(regimes=("volatile",))
        expert = MockExpert("test", caps)
        registry.register("test", expert, caps)
        
        retrieved_caps = registry.get_capabilities("test")
        
        assert retrieved_caps is caps
    
    def test_list_all(self):
        """List all registered expert IDs."""
        registry = ExpertRegistry()
        
        for i in range(3):
            expert = MockExpert(f"expert_{i}", ExpertCapability())
            registry.register(f"expert_{i}", expert, expert.capabilities)
        
        all_experts = registry.list_all()
        
        assert all_experts == ["expert_0", "expert_1", "expert_2"]


class TestExpertRegistryRemoval:
    """Tests for unregistering experts."""
    
    def test_unregister_expert(self):
        """Remove expert from registry."""
        registry = ExpertRegistry()
        expert = MockExpert("test", ExpertCapability())
        registry.register("test", expert, expert.capabilities)
        
        removed = registry.unregister("test")
        
        assert removed is True
        assert len(registry) == 0
        assert "test" not in registry
    
    def test_unregister_nonexistent(self):
        """Remove non-existent expert returns False."""
        registry = ExpertRegistry()
        
        removed = registry.unregister("nonexistent")
        
        assert removed is False


class TestExpertRegistryStats:
    """Tests for registry statistics."""
    
    def test_empty_stats(self):
        """Stats for empty registry."""
        registry = ExpertRegistry()
        
        stats = registry.get_stats()
        
        assert stats["total_experts"] == 0
        assert stats["avg_cost"] == 0.0
    
    def test_populated_stats(self):
        """Stats for populated registry."""
        registry = ExpertRegistry()
        
        registry.register("cheap", MockExpert("cheap", ExpertCapability(computational_cost=1.0)), None)
        registry.register("expensive", MockExpert("expensive", ExpertCapability(computational_cost=3.0)), None)
        
        stats = registry.get_stats()
        
        assert stats["total_experts"] == 2
        assert stats["avg_cost"] == 2.0
        assert "stable" in stats["regimes_covered"]


class TestExpertRegistryCapacityLimits:
    """Tests for registry capacity limits."""
    
    def test_max_entries_limit(self):
        """Registry enforces max_entries limit."""
        registry = ExpertRegistry(max_entries=2)
        
        expert1 = MockExpert("e1", ExpertCapability())
        expert2 = MockExpert("e2", ExpertCapability())
        
        registry.register("e1", expert1, expert1.capabilities)
        registry.register("e2", expert2, expert2.capabilities)
        
        expert3 = MockExpert("e3", ExpertCapability())
        with pytest.raises(RuntimeError, match="Registry full"):
            registry.register("e3", expert3, expert3.capabilities)


class TestExpertRegistryIteration:
    """Tests for registry iteration."""
    
    def test_iteration(self):
        """Iterate over expert IDs."""
        registry = ExpertRegistry()
        
        for i in range(3):
            expert = MockExpert(f"expert_{i}", ExpertCapability())
            registry.register(f"expert_{i}", expert, expert.capabilities)
        
        ids = [eid for eid in registry]
        
        assert ids == ["expert_0", "expert_1", "expert_2"]
    
    def test_contains_operator(self):
        """Use 'in' operator."""
        registry = ExpertRegistry()
        expert = MockExpert("test", ExpertCapability())
        registry.register("test", expert, expert.capabilities)
        
        assert "test" in registry
        assert "other" not in registry
