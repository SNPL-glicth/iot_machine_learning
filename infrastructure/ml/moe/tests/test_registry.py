"""Tests para ExpertRegistry.

Cobertura objetivo: 80%+ de líneas.
"""

import pytest
from unittest.mock import MagicMock

from iot_machine_learning.domain.ports.expert_port import ExpertCapability
from ..registry import ExpertRegistry, ExpertEntry


class MockExpert:
    """Mock experto para testing."""
    
    def __init__(self, name):
        self._name = name
        self._caps = ExpertCapability(
            regimes=("stable",),
            domains=("iot",),
            min_points=3,
            computational_cost=1.0,
        )
    
    @property
    def name(self):
        return self._name
    
    @property
    def capabilities(self):
        return self._caps
    
    def predict(self, window):
        pass
    
    def can_handle(self, window):
        return True


class TestExpertRegistryBasics:
    """Tests básicos de registro y recuperación."""
    
    def test_register_single_expert(self):
        registry = ExpertRegistry()
        expert = MockExpert("test_expert")
        
        registry.register("test_expert", expert, expert.capabilities)
        
        assert len(registry) == 1
        assert "test_expert" in registry
    
    def test_register_multiple_experts(self):
        registry = ExpertRegistry()
        
        for i in range(5):
            expert = MockExpert(f"expert_{i}")
            registry.register(f"expert_{i}", expert, expert.capabilities)
        
        assert len(registry) == 5
        assert registry.list_all() == ["expert_0", "expert_1", "expert_2", "expert_3", "expert_4"]
    
    def test_get_expert(self):
        registry = ExpertRegistry()
        expert = MockExpert("test_expert")
        
        registry.register("test_expert", expert, expert.capabilities)
        retrieved = registry.get("test_expert")
        
        assert retrieved is expert
    
    def test_get_nonexistent_expert(self):
        registry = ExpertRegistry()
        
        result = registry.get("nonexistent")
        
        assert result is None
    
    def test_unregister_expert(self):
        registry = ExpertRegistry()
        expert = MockExpert("test_expert")
        
        registry.register("test_expert", expert, expert.capabilities)
        removed = registry.unregister("test_expert")
        
        assert removed is True
        assert len(registry) == 0
    
    def test_unregister_nonexistent(self):
        registry = ExpertRegistry()
        
        removed = registry.unregister("nonexistent")
        
        assert removed is False
    
    def test_duplicate_registration_raises(self):
        registry = ExpertRegistry()
        expert = MockExpert("test_expert")
        
        registry.register("test_expert", expert, expert.capabilities)
        
        with pytest.raises(ValueError) as exc_info:
            registry.register("test_expert", expert, expert.capabilities)
        
        assert "already registered" in str(exc_info.value)
    
    def test_force_overwrite(self):
        registry = ExpertRegistry()
        expert1 = MockExpert("test_expert")
        expert2 = MockExpert("test_expert")
        
        registry.register("test_expert", expert1, expert1.capabilities)
        registry.register("test_expert", expert2, expert2.capabilities, force=True)
        
        retrieved = registry.get("test_expert")
        assert retrieved is expert2


class TestExpertRegistrySearch:
    """Tests de búsqueda por capacidades."""
    
    @pytest.fixture
    def populated_registry(self):
        registry = ExpertRegistry()
        
        # Experto estable
        stable = MockExpert("stable_expert")
        stable._caps = ExpertCapability(regimes=("stable",), min_points=3)
        registry.register("stable_expert", stable, stable._caps)
        
        # Experto volátil
        volatile = MockExpert("volatile_expert")
        volatile._caps = ExpertCapability(regimes=("volatile",), min_points=5)
        registry.register("volatile_expert", volatile, volatile._caps)
        
        # Experto trending
        trending = MockExpert("trending_expert")
        trending._caps = ExpertCapability(regimes=("trending",), min_points=5)
        registry.register("trending_expert", trending, trending._caps)
        
        # Experto multi-régimen
        multi = MockExpert("multi_expert")
        multi._caps = ExpertCapability(regimes=("stable", "trending"), min_points=3)
        registry.register("multi_expert", multi, multi._caps)
        
        return registry
    
    def test_find_by_regime(self, populated_registry):
        results = populated_registry.find_by_regime("stable")
        
        assert "stable_expert" in results
        assert "multi_expert" in results
        assert "volatile_expert" not in results
    
    def test_find_by_regime_no_matches(self, populated_registry):
        results = populated_registry.find_by_regime("noisy")
        
        assert results == []
    
    def test_get_candidates_by_points(self, populated_registry):
        from domain.model.context_vector import ContextVector
        context = ContextVector(
            regime="stable",
            domain="iot",
            n_points=4,
            signal_features={}
        )
        candidates = populated_registry.get_candidates(context)
        
        assert "stable_expert" in candidates
        assert "multi_expert" in candidates
        assert "volatile_expert" not in candidates  # Requiere 5 puntos
    
    def test_get_candidates_by_regime(self, populated_registry):
        from domain.model.context_vector import ContextVector
        context = ContextVector(
            regime="trending",
            domain="iot",
            n_points=5,
            signal_features={}
        )
        candidates = populated_registry.get_candidates(context)
        
        assert "trending_expert" in candidates
        assert "multi_expert" in candidates
        assert "stable_expert" not in candidates


class TestExpertRegistryLimits:
    """Tests de límites y edge cases."""
    
    def test_max_entries_limit(self):
        registry = ExpertRegistry(max_entries=3)
        
        for i in range(3):
            expert = MockExpert(f"expert_{i}")
            registry.register(f"expert_{i}", expert, expert.capabilities)
        
        with pytest.raises(RuntimeError) as exc_info:
            expert = MockExpert("expert_overflow")
            registry.register("expert_overflow", expert, expert.capabilities)
        
        assert "Registry full" in str(exc_info.value)
    
    def test_iteration(self):
        registry = ExpertRegistry()
        
        for i in range(3):
            expert = MockExpert(f"expert_{i}")
            registry.register(f"expert_{i}", expert, expert.capabilities)
        
        ids = list(registry)
        
        assert ids == ["expert_0", "expert_1", "expert_2"]
    
    def test_stats_empty(self):
        registry = ExpertRegistry()
        
        stats = registry.get_stats()
        
        assert stats["total_experts"] == 0
        assert stats["avg_cost"] == 0.0
    
    def test_stats_populated(self):
        registry = ExpertRegistry()
        
        expert1 = MockExpert("expert1")
        expert1._caps = ExpertCapability(computational_cost=1.0)
        registry.register("expert1", expert1, expert1._caps)
        
        expert2 = MockExpert("expert2")
        expert2._caps = ExpertCapability(computational_cost=2.0)
        registry.register("expert2", expert2, expert2._caps)
        
        stats = registry.get_stats()
        
        assert stats["total_experts"] == 2
        assert stats["avg_cost"] == 1.5


class TestExpertRegistryThreading:
    """Tests de thread-safety (básicos)."""
    
    def test_concurrent_reads(self):
        import threading
        
        registry = ExpertRegistry()
        expert = MockExpert("test")
        registry.register("test", expert, expert.capabilities)
        
        results = []
        
        def read_registry():
            for _ in range(100):
                results.append(registry.get("test"))
        
        threads = [threading.Thread(target=read_registry) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(results) == 500
        assert all(r is expert for r in results)
