"""Thread Safety Tests for Plasticity Components.

Tests concurrent access to ContextualPlasticityTracker and EngineHealthMonitor
to ensure RLock protection prevents race conditions.
"""

import threading
import time
import pytest
from datetime import datetime

from iot_machine_learning.infrastructure.ml.cognitive.plasticity.contextual_plasticity_tracker import ContextualPlasticityTracker
from iot_machine_learning.infrastructure.ml.cognitive.monitoring.engine_health_monitor import EngineHealthMonitor
from iot_machine_learning.domain.entities.plasticity.plasticity_context import PlasticityContext, RegimeType


class TestContextualTrackerThreadSafety:
    """Test thread safety of ContextualPlasticityTracker."""
    
    def test_concurrent_writes_no_race_condition(self):
        """10 threads writing 100 errors each should result in ~50 (window limit)."""
        tracker = ContextualPlasticityTracker(window_size=50)
        
        ctx = PlasticityContext(
            regime=RegimeType.STABLE,
            noise_ratio=0.1,
            volatility=0.2,
            time_of_day=10,
            consecutive_failures=0,
            error_magnitude=0.0,
            is_critical_zone=False,
            timestamp=datetime.now(),
        )
        
        def worker():
            for i in range(100):
                tracker.record_error("test_series", "engine1", float(i % 10), ctx)
        
        # Launch 10 threads
        threads = [threading.Thread(target=worker) for _ in range(10)]
        start = time.time()
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        elapsed = time.time() - start
        
        # Verify no crash and reasonable execution time
        assert elapsed < 5.0, f"Took too long: {elapsed:.2f}s (possible deadlock)"
        
        # Verify window size is respected (should be ~50, not 1000)
        weights = tracker.get_contextual_weights("test_series", ["engine1"], ctx)
        assert weights is not None, "Weights should be computed"
        assert weights.get("engine1", 0.0) > 0.0, "Weight should be positive"
        
        # Check internal state
        context_key = ctx.context_key
        errors = tracker._errors.get("test_series", {}).get("engine1", {}).get(context_key, [])
        assert len(errors) <= 50, f"Expected ≤50 errors, got {len(errors)}"
        assert len(errors) >= 40, f"Expected ≥40 errors (some writes), got {len(errors)}"
    
    def test_concurrent_reads_writes_no_deadlock(self):
        """Mix of reads and writes should not deadlock."""
        tracker = ContextualPlasticityTracker()
        
        ctx = PlasticityContext(
            regime=RegimeType.VOLATILE,
            noise_ratio=0.3,
            volatility=0.7,
            time_of_day=15,
            consecutive_failures=0,
            error_magnitude=0.0,
            is_critical_zone=False,
            timestamp=datetime.now(),
        )
        
        # Pre-populate with some data
        for i in range(10):
            tracker.record_error("test_series", "engine1", float(i), ctx)
        
        results = []
        
        def writer():
            for i in range(50):
                tracker.record_error("test_series", "engine1", float(i), ctx)
        
        def reader():
            for _ in range(50):
                weights = tracker.get_contextual_weights("test_series", ["engine1"], ctx)
                if weights:
                    results.append(weights.get("engine1", 0.0))
        
        # 5 writers + 5 readers
        threads = []
        threads.extend([threading.Thread(target=writer) for _ in range(5)])
        threads.extend([threading.Thread(target=reader) for _ in range(5)])
        
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - start
        
        # Should complete quickly without deadlock
        assert elapsed < 5.0, f"Took too long: {elapsed:.2f}s (possible deadlock)"
        
        # Most reads should have succeeded (some may return None if insufficient samples)
        assert len(results) >= 100, f"Expected ≥100 successful reads, got {len(results)}"
        assert all(w > 0 for w in results), "All weights should be positive"
    
    def test_concurrent_reset_no_crash(self):
        """Concurrent resets should not crash."""
        tracker = ContextualPlasticityTracker()
        
        ctx = PlasticityContext(
            regime=RegimeType.STABLE,
            noise_ratio=0.1,
            volatility=0.2,
            time_of_day=10,
            consecutive_failures=0,
            error_magnitude=0.0,
            is_critical_zone=False,
            timestamp=datetime.now(),
        )
        
        def worker():
            for i in range(20):
                tracker.record_error(f"series_{i % 3}", "engine1", float(i), ctx)
                if i % 5 == 0:
                    tracker.reset(series_id=f"series_{i % 3}")
        
        threads = [threading.Thread(target=worker) for _ in range(5)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should complete without crash
        # State may be inconsistent but should not crash


class TestHealthMonitorThreadSafety:
    """Test thread safety of EngineHealthMonitor."""
    
    def test_concurrent_predictions_no_race_condition(self):
        """10 threads recording predictions should maintain consistent state."""
        monitor = EngineHealthMonitor(failure_threshold=5, error_tolerance=1.0)
        
        def worker():
            for i in range(20):
                # Alternate between success and failure
                error = 0.5 if i % 2 == 0 else 5.0
                monitor.record_prediction("test_series", "engine1", error)
        
        threads = [threading.Thread(target=worker) for _ in range(10)]
        start = time.time()
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        elapsed = time.time() - start
        
        # Should complete quickly
        assert elapsed < 5.0, f"Took too long: {elapsed:.2f}s (possible deadlock)"
        
        # Verify state exists and is consistent
        state = monitor.get_state("test_series", "engine1")
        assert state is not None, "State should exist"
        assert state.total_predictions == 200, f"Expected 200 predictions, got {state.total_predictions}"
        
        # Consecutive failures should be reasonable (not corrupted)
        assert 0 <= state.consecutive_failures <= 20, f"Consecutive failures out of range: {state.consecutive_failures}"
    
    def test_concurrent_inhibition_checks_no_deadlock(self):
        """Mix of predictions and inhibition checks should not deadlock."""
        monitor = EngineHealthMonitor(failure_threshold=3)
        
        inhibition_results = []
        
        def predictor():
            for i in range(30):
                error = 10.0  # Always fail
                monitor.record_prediction("test_series", "engine1", error)
        
        def checker():
            for _ in range(30):
                is_inhibited = monitor.is_inhibited("test_series", "engine1")
                inhibition_results.append(is_inhibited)
        
        # 3 predictors + 3 checkers
        threads = []
        threads.extend([threading.Thread(target=predictor) for _ in range(3)])
        threads.extend([threading.Thread(target=checker) for _ in range(3)])
        
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - start
        
        # Should complete quickly
        assert elapsed < 5.0, f"Took too long: {elapsed:.2f}s (possible deadlock)"
        
        # All checks should have completed
        assert len(inhibition_results) == 90, f"Expected 90 checks, got {len(inhibition_results)}"
        
        # Eventually should become inhibited (after 3 failures)
        assert any(inhibition_results), "Engine should have been inhibited at some point"
    
    def test_concurrent_multiple_engines_no_corruption(self):
        """Multiple threads updating different engines should not corrupt state."""
        monitor = EngineHealthMonitor(failure_threshold=5)
        
        def worker(engine_name: str):
            for i in range(50):
                error = float(i % 3)  # Varying errors
                monitor.record_prediction("test_series", engine_name, error)
        
        # 5 engines, 2 threads each
        threads = []
        for engine_num in range(5):
            engine_name = f"engine_{engine_num}"
            threads.extend([threading.Thread(target=worker, args=(engine_name,)) for _ in range(2)])
        
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - start
        
        # Should complete quickly
        assert elapsed < 5.0, f"Took too long: {elapsed:.2f}s (possible deadlock)"
        
        # Verify all engines have correct prediction counts
        summary = monitor.get_health_summary("test_series")
        assert len(summary) == 5, f"Expected 5 engines, got {len(summary)}"
        
        for engine_name, metrics in summary.items():
            assert metrics["total_predictions"] == 100, f"{engine_name}: expected 100 predictions, got {metrics['total_predictions']}"
    
    def test_concurrent_reset_no_crash(self):
        """Concurrent resets should not crash."""
        monitor = EngineHealthMonitor()
        
        def worker():
            for i in range(20):
                monitor.record_prediction(f"series_{i % 3}", "engine1", float(i))
                if i % 5 == 0:
                    monitor.reset(series_id=f"series_{i % 3}")
        
        threads = [threading.Thread(target=worker) for _ in range(5)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should complete without crash


class TestIntegratedThreadSafety:
    """Test both components working together under concurrency."""
    
    def test_tracker_and_monitor_concurrent_no_interference(self):
        """Both components used concurrently should not interfere."""
        tracker = ContextualPlasticityTracker()
        monitor = EngineHealthMonitor(failure_threshold=10)
        
        ctx = PlasticityContext(
            regime=RegimeType.STABLE,
            noise_ratio=0.1,
            volatility=0.2,
            time_of_day=10,
            consecutive_failures=0,
            error_magnitude=0.0,
            is_critical_zone=False,
            timestamp=datetime.now(),
        )
        
        def worker():
            for i in range(50):
                error = float(i % 5)
                
                # Update both components
                tracker.record_error("test_series", "engine1", error, ctx)
                monitor.record_prediction("test_series", "engine1", error)
        
        threads = [threading.Thread(target=worker) for _ in range(5)]
        start = time.time()
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        elapsed = time.time() - start
        
        # Should complete quickly
        assert elapsed < 5.0, f"Took too long: {elapsed:.2f}s (possible deadlock)"
        
        # Verify both components have data
        weights = tracker.get_contextual_weights("test_series", ["engine1"], ctx)
        assert weights is not None, "Tracker should have computed weights"
        assert weights.get("engine1", 0.0) > 0.0, "Weight should be positive"
        
        state = monitor.get_state("test_series", "engine1")
        assert state is not None, "Monitor should have state"
        assert state.total_predictions == 250, f"Expected 250 predictions, got {state.total_predictions}"


class TestPerformanceOverhead:
    """Test that thread safety doesn't add significant overhead."""
    
    def test_single_thread_performance_baseline(self):
        """Measure baseline performance without contention."""
        tracker = ContextualPlasticityTracker()
        
        ctx = PlasticityContext(
            regime=RegimeType.STABLE,
            noise_ratio=0.1,
            volatility=0.2,
            time_of_day=10,
            consecutive_failures=0,
            error_magnitude=0.0,
            is_critical_zone=False,
            timestamp=datetime.now(),
        )
        
        start = time.time()
        for i in range(1000):
            tracker.record_error("test_series", "engine1", float(i % 10), ctx)
        elapsed = time.time() - start
        
        # Should be very fast (< 100ms for 1000 operations)
        assert elapsed < 0.1, f"Single thread too slow: {elapsed:.3f}s"
    
    def test_multi_thread_performance_acceptable(self):
        """Measure performance with 10 threads (should be < 5x slower)."""
        tracker = ContextualPlasticityTracker()
        
        ctx = PlasticityContext(
            regime=RegimeType.STABLE,
            noise_ratio=0.1,
            volatility=0.2,
            time_of_day=10,
            consecutive_failures=0,
            error_magnitude=0.0,
            is_critical_zone=False,
            timestamp=datetime.now(),
        )
        
        def worker():
            for i in range(100):
                tracker.record_error("test_series", "engine1", float(i % 10), ctx)
        
        threads = [threading.Thread(target=worker) for _ in range(10)]
        
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 500ms for 1000 operations across 10 threads)
        assert elapsed < 0.5, f"Multi-thread too slow: {elapsed:.3f}s (possible lock contention)"
