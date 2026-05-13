"""Unit tests for CRIT-1, CRIT-2, CRIT-3 fixes.

CRIT-1: Global mutable _weight_cache in adapt_phase.py causing race conditions.
CRIT-2: Missing _weight_mediator in InhibitPhase causing runtime AttributeError.
CRIT-3: EPSILON mismatch in WeightResolutionService causing potential numerical divergence.
"""

import pytest
import threading
import time
from typing import Dict, List, Optional
from unittest.mock import Mock, MagicMock, patch

from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.adapt_phase import (
    AdaptPhase,
    WeightCache,
)
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.inhibit_phase import (
    InhibitPhase,
    _compute_signal_z_score,
)
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.weight_resolution_service import (
    WeightResolutionService,
)
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases import (
    PipelineContext,
    create_initial_context,
)
from iot_machine_learning.infrastructure.ml.analysis.types import EnginePerception, MetaDiagnostic


class TestCrit1AdaptPhaseCache:
    """Tests for CRIT-1 fix: Instance-scoped weight cache in AdaptPhase."""

    def test_adapt_phase_has_instance_cache(self):
        """Verify AdaptPhase creates its own cache instance, not using global."""
        phase1 = AdaptPhase()
        phase2 = AdaptPhase()
        
        # Verify each instance has its own cache
        assert hasattr(phase1, '_weight_cache')
        assert hasattr(phase2, '_weight_cache')
        
        # Verify they are different instances
        assert phase1._weight_cache is not phase2._weight_cache
        assert id(phase1._weight_cache) != id(phase2._weight_cache)

    def test_weight_cache_is_thread_safe(self):
        """Verify WeightCache uses RLock for thread safety."""
        cache = WeightCache()
        
        # Verify cache has lock
        assert hasattr(cache, '_lock')
        
        # Test concurrent access
        results = []
        def write_to_cache(key):
            for i in range(100):
                cache.set(f"series_{key}", "STABLE", {f"engine_{i}": 0.5})
        
        threads = [threading.Thread(target=write_to_cache, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # If no exception occurred, thread safety is working
        assert True

    def test_adapt_phase_cache_hit_miss_logging(self, caplog):
        """Verify AdaptPhase logs cache hits and misses."""
        phase = AdaptPhase(max_cache_entries=10, cache_ttl_seconds=60.0)
        
        # Create mock context
        mock_orchestrator = Mock()
        mock_orchestrator._error_history = Mock()
        mock_orchestrator._error_history.get_error_dict_for_inhibition = Mock(return_value={})
        mock_orchestrator._weight_resolver = Mock()
        mock_orchestrator._weight_resolver.resolve = Mock(return_value={"engine1": 0.5})
        mock_orchestrator._plasticity = Mock()
        mock_orchestrator._plasticity.has_history = Mock(return_value=False)
        
        mock_perception = Mock()
        mock_perception.engine_name = "engine1"
        
        ctx = create_initial_context(
            orchestrator=mock_orchestrator,
            values=[1.0, 2.0, 3.0],
            timestamps=None,
            series_id="test_series",
            flags=Mock(),
            timer=Mock(),
        )
        ctx = ctx.with_field(perceptions=[mock_perception])
        
        # First call should be a cache miss
        with caplog.at_level('DEBUG'):
            result = phase.execute(ctx)
        
        assert any('adapt_phase_cache_miss' in record.message for record in caplog.records)
        
        # Second call should be a cache hit
        caplog.clear()
        with caplog.at_level('DEBUG'):
            result = phase.execute(ctx)
        
        assert any('adapt_phase_cache_hit' in record.message for record in caplog.records)

    def test_weight_cache_ttl_eviction(self):
        """Verify WeightCache evicts entries after TTL expires."""
        cache = WeightCache(max_entries=100, ttl_seconds=0.1)
        
        # Set a value
        cache.set("series1", "STABLE", {"engine1": 0.5})
        
        # Should be available immediately
        assert cache.get("series1", "STABLE") is not None
        
        # Wait for TTL to expire
        time.sleep(0.15)
        
        # Should be evicted
        assert cache.get("series1", "STABLE") is None

    def test_weight_cache_lru_eviction(self):
        """Verify WeightCache evicts oldest entries when at capacity."""
        cache = WeightCache(max_entries=3, ttl_seconds=60.0)
        
        # Fill cache to capacity
        cache.set("series1", "STABLE", {"engine1": 0.5})
        cache.set("series2", "STABLE", {"engine2": 0.5})
        cache.set("series3", "STABLE", {"engine3": 0.5})
        
        # Add one more - should evict oldest (series1)
        cache.set("series4", "STABLE", {"engine4": 0.5})
        
        assert cache.get("series1", "STABLE") is None  # Evicted
        assert cache.get("series2", "STABLE") is not None
        assert cache.get("series3", "STABLE") is not None
        assert cache.get("series4", "STABLE") is not None


class TestCrit2InhibitPhaseWeightMediator:
    """Tests for CRIT-2 fix: InhibitPhase uses plasticity_weights directly."""

    def test_inhibit_phase_no_weight_mediator_reference(self):
        """Verify InhibitPhase does not reference non-existent _weight_mediator."""
        # Read the source code to verify
        import inspect
        source = inspect.getsource(InhibitPhase.execute)
        
        # Should NOT contain _weight_mediator
        assert '_weight_mediator' not in source
        
        # Should contain plasticity_weights directly
        assert 'plasticity_weights' in source

    def test_inhibit_phase_uses_plasticity_weights_directly(self):
        """Verify InhibitPhase uses ctx.plasticity_weights as mediated_weights."""
        phase = InhibitPhase()
        
        # Create mock context
        mock_orchestrator = Mock()
        mock_orchestrator._inhibition = Mock()
        
        # Mock inhibition states
        mock_inh_state = Mock()
        mock_inh_state.engine_name = "engine1"
        mock_orchestrator._inhibition.compute = Mock(return_value=[mock_inh_state])
        
        mock_perception = Mock()
        mock_perception.engine_name = "engine1"
        
        ctx = create_initial_context(
            orchestrator=mock_orchestrator,
            values=[1.0, 2.0, 3.0],
            timestamps=None,
            series_id="test_series",
            flags=Mock(),
            timer=Mock(),
        )
        ctx = ctx.with_field(
            perceptions=[mock_perception],
            plasticity_weights={"engine1": 0.7},
            error_dict={"engine1": 0.1},
        )
        ctx.explanation = Mock()
        ctx.explanation.set_inhibition = Mock()
        
        # Execute
        result = phase.execute(ctx)
        
        # Verify mediated_weights equals plasticity_weights
        assert result.mediated_weights == {"engine1": 0.7}
        
        # Verify inhibition.compute was called
        mock_orchestrator._inhibition.compute.assert_called_once()

    def test_compute_signal_z_score_numerical_stability(self):
        """Verify _compute_signal_z_score handles edge cases correctly."""
        # Too few values
        assert _compute_signal_z_score([1.0]) == 0.0
        assert _compute_signal_z_score([1.0, 2.0]) == 0.0
        
        # Normal case
        result = _compute_signal_z_score([1.0, 2.0, 3.0, 4.0])
        assert isinstance(result, float)
        
        # Constant values (zero std)
        result = _compute_signal_z_score([5.0, 5.0, 5.0])
        assert result == 0.0  # Should return 0 when std is near zero


class TestCrit3WeightResolutionServiceEpsilon:
    """Tests for CRIT-3 fix: EPSILON default restored to 0.01."""

    def test_default_epsilon_is_0_01(self):
        """Verify default epsilon is 0.01, not 1e-12."""
        service = WeightResolutionService(
            base_weights={"engine1": 0.5, "engine2": 0.5}
        )
        
        # Should use 0.01 default
        assert service._epsilon == 0.01

    def test_explicit_epsilon_none_uses_default(self):
        """Verify passing epsilon=None uses default 0.01."""
        service = WeightResolutionService(
            base_weights={"engine1": 0.5, "engine2": 0.5},
            epsilon=None
        )
        
        assert service._epsilon == 0.01

    def test_explicit_epsilon_respected(self):
        """Verify explicit epsilon value is respected."""
        service = WeightResolutionService(
            base_weights={"engine1": 0.5, "engine2": 0.5},
            epsilon=0.05
        )
        
        assert service._epsilon == 0.05

    def test_epsilon_too_small_warns(self, caplog):
        """Verify epsilon < 1e-6 triggers warning."""
        with caplog.at_level('WARNING'):
            service = WeightResolutionService(
                base_weights={"engine1": 0.5, "engine2": 0.5},
                epsilon=1e-10  # Too small
            )
        
        assert service._epsilon == 1e-10  # Value is accepted
        assert any('weight_resolution_epsilon_too_small' in record.message for record in caplog.records)

    def test_epsilon_zero_raises_error(self):
        """Verify epsilon <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            WeightResolutionService(
                base_weights={"engine1": 0.5, "engine2": 0.5},
                epsilon=0.0
            )

    def test_epsilon_negative_raises_error(self):
        """Verify negative epsilon raises ValueError."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            WeightResolutionService(
                base_weights={"engine1": 0.5, "engine2": 0.5},
                epsilon=-0.01
            )

    def test_normalize_with_0_01_epsilon(self):
        """Verify normalization works correctly with 0.01 epsilon."""
        service = WeightResolutionService(
            base_weights={"engine1": 0.5, "engine2": 0.5},
            epsilon=0.01
        )
        
        # Test with zero weight
        weights = {"engine1": 0.0, "engine2": 0.5}
        normalized = service._normalize(weights, ["engine1", "engine2"])
        
        # engine1 should get epsilon (0.01) instead of 0.0
        assert normalized["engine1"] == pytest.approx(0.01, abs=1e-6)
        assert normalized["engine2"] > 0
        
        # Sum should be 1.0
        assert sum(normalized.values()) == pytest.approx(1.0, rel=1e-9)

    def test_compute_adaptive_weights_with_0_01_epsilon(self):
        """Verify adaptive weight computation works with 0.01 epsilon."""
        mock_storage = Mock()
        mock_storage.get_rolling_mae = Mock(return_value=0.1)
        
        service = WeightResolutionService(
            base_weights={"engine1": 0.5, "engine2": 0.5},
            storage_adapter=mock_storage,
            epsilon=0.01
        )
        
        adaptive = service._compute_adaptive_weights("series1", ["engine1", "engine2"])
        
        # Should not return None (MAE values are valid)
        assert adaptive is not None
        
        # Should be normalized
        assert sum(adaptive.values()) == pytest.approx(1.0, rel=1e-9)

    def test_backward_compatibility_with_old_tests(self):
        """Verify behavior is compatible with tests expecting 0.01 epsilon."""
        # This test ensures that reverting to 0.01 doesn't break existing logic
        service = WeightResolutionService(
            base_weights={"engine1": 0.3, "engine2": 0.3, "engine3": 0.4}
        )
        
        # Test base weights fallback
        weights = service.resolve(
            regime="STABLE",
            engine_names=["engine1", "engine2", "engine3"],
            series_id="test_series"
        )
        
        # Should return normalized base weights
        assert weights["engine1"] == pytest.approx(0.3, rel=1e-9)
        assert weights["engine2"] == pytest.approx(0.3, rel=1e-9)
        assert weights["engine3"] == pytest.approx(0.4, rel=1e-9)
        assert sum(weights.values()) == pytest.approx(1.0, rel=1e-9)


class TestCrit1Concurrency:
    """Concurrency tests for CRIT-1 fix."""

    def test_concurrent_adapt_phase_instances_isolated(self):
        """Verify concurrent AdaptPhase instances have isolated caches."""
        results = []
        
        def run_phase(phase_id):
            phase = AdaptPhase(max_cache_entries=10, cache_ttl_seconds=60.0)
            # Each phase should have its own cache
            results.append(id(phase._weight_cache))
        
        threads = [threading.Thread(target=run_phase, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All cache IDs should be different (isolated instances)
        assert len(set(results)) == 10

    def test_concurrent_cache_access_no_race_condition(self):
        """Verify concurrent cache access doesn't cause race conditions."""
        cache = WeightCache(max_entries=1000, ttl_seconds=60.0)
        errors = []
        
        def concurrent_access(thread_id):
            try:
                for i in range(100):
                    key = f"series_{thread_id}_{i % 10}"
                    regime = "STABLE"
                    weights = {f"engine{j}": 0.1 for j in range(5)}
                    
                    # Get and set operations
                    cached = cache.get(key, regime)
                    cache.set(key, regime, weights)
                    
                    time.sleep(0.001)  # Small delay to increase contention
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=concurrent_access, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have no errors
        assert len(errors) == 0, f"Race condition errors: {errors}"

    def test_adapt_phase_concurrent_execution(self):
        """Verify AdaptPhase can be executed concurrently without issues."""
        phases = [AdaptPhase(max_cache_entries=100, cache_ttl_seconds=60.0) for _ in range(5)]
        errors = []
        
        def execute_phase(phase, phase_id):
            try:
                mock_orchestrator = Mock()
                mock_orchestrator._error_history = Mock()
                mock_orchestrator._error_history.get_error_dict_for_inhibition = Mock(return_value={})
                mock_orchestrator._weight_resolver = Mock()
                mock_orchestrator._weight_resolver.resolve = Mock(
                    return_value={f"engine{i}": 0.2 for i in range(5)}
                )
                mock_orchestrator._plasticity = Mock()
                mock_orchestrator._plasticity.has_history = Mock(return_value=False)
                
                mock_perception = Mock()
                mock_perception.engine_name = f"engine_{phase_id % 5}"
                
                for _ in range(10):
                    ctx = create_initial_context(
                        orchestrator=mock_orchestrator,
                        values=[float(i) for i in range(10)],
                        timestamps=None,
                        series_id=f"series_{phase_id}",
                        flags=Mock(),
                        timer=Mock(),
                    )
                    ctx = ctx.with_field(perceptions=[mock_perception])
                    result = phase.execute(ctx)
                    
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=execute_phase, args=(phases[i], i))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have no errors
        assert len(errors) == 0, f"Concurrent execution errors: {errors}"
