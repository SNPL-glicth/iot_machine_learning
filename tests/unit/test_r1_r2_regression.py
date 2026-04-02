"""Tests for R-1 Zero-Leakage and R-2 Smart Inhibition refactor.

Validates:
1. Per-series state isolation (no cross-contamination)
2. Smart inhibition with z-score context
3. Adaptive alpha values per regime
"""

import pytest
import threading
import time
from typing import List

from iot_machine_learning.infrastructure.ml.cognitive.orchestration.context_state_manager import (
    ContextStateManager,
    SeriesState,
)
from iot_machine_learning.infrastructure.ml.cognitive.inhibition.smart_rules import (
    check_failure_threshold,
    evaluate_inhibition,
    _ANOMALY_Z_SCORE_THRESHOLD,
)
from iot_machine_learning.ml_service.metrics.prometheus_exporter import (
    PrometheusExporter,
    get_exporter,
    reset_exporter,
)


class MockState:
    """Mock state for inhibition testing."""
    
    def __init__(
        self,
        series_id: str,
        engine_name: str,
        consecutive_failures: int = 0,
        is_inhibited: bool = False,
    ):
        self.series_id = series_id
        self.engine_name = engine_name
        self.consecutive_failures = consecutive_failures
        self.is_inhibited = is_inhibited
        self.inhibition_reason = None
        self.hours_since_last_success = None
    
    def with_inhibition(self, reason: str):
        new = MockState(
            self.series_id,
            self.engine_name,
            self.consecutive_failures,
            True,
        )
        new.inhibition_reason = reason
        return new


class TestContextStateIsolation:
    """R-1: Verify zero leakage between series."""
    
    def test_series_state_isolation(self):
        """Sensor A state must not affect Sensor B."""
        manager = ContextStateManager()
        
        # Set state for sensor_1
        manager.update_regime("sensor_1", "VOLATILE")
        manager.update_perceptions("sensor_1", [])
        
        # Get state for sensor_2
        state_2 = manager.get_state("sensor_2")
        
        # sensor_2 must have default values, not sensor_1's
        assert state_2.last_regime is None
        assert state_2.last_perceptions == []
    
    def test_per_series_regime_tracking(self):
        """Each series tracks its own regime."""
        manager = ContextStateManager()
        
        manager.update_regime("series_a", "STABLE")
        manager.update_regime("series_b", "VOLATILE")
        
        assert manager.get_regime("series_a") == "STABLE"
        assert manager.get_regime("series_b") == "VOLATILE"
    
    def test_thread_safety_isolation(self):
        """Concurrent access doesn't cause cross-contamination."""
        manager = ContextStateManager()
        errors: List[Exception] = []
        
        def worker(series_id: str, regime: str):
            try:
                for _ in range(100):
                    manager.update_regime(series_id, regime)
                    time.sleep(0.001)
                    actual = manager.get_regime(series_id)
                    if actual != regime:
                        errors.append(
                            ValueError(f"Expected {regime}, got {actual}")
                        )
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=worker, args=("sensor_1", "STABLE")),
            threading.Thread(target=worker, args=("sensor_2", "VOLATILE")),
            threading.Thread(target=worker, args=("sensor_3", "NOISY")),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert not errors, f"Thread safety errors: {errors}"
    
    def test_eviction_doesnt_affect_others(self):
        """LRU eviction of one series doesn't corrupt others."""
        manager = ContextStateManager(max_series=3)
        
        # Add 3 series
        manager.update_regime("s1", "A")
        manager.update_regime("s2", "B")
        manager.update_regime("s3", "C")
        
        # Add 4th (evicts oldest)
        manager.update_regime("s4", "D")
        
        # Verify s2, s3, s4 are intact
        assert manager.get_regime("s2") == "B"
        assert manager.get_regime("s3") == "C"
        assert manager.get_regime("s4") == "D"


class TestSmartInhibition:
    """R-2: Smart inhibition with z-score context."""
    
    def test_normal_inhibition_threshold(self):
        """Without anomaly context, threshold is standard."""
        state = MockState("s1", "engine", consecutive_failures=3)
        
        reason = check_failure_threshold(state, failure_threshold=3)
        
        assert reason is not None
        assert "Consecutive failures" in reason
    
    def test_smart_inhibition_raises_threshold(self):
        """With high z-score, threshold is raised (harder to inhibit)."""
        state = MockState("s1", "engine", consecutive_failures=5)
        
        # Normal: 5 failures > threshold=3 → inhibited
        reason_normal = check_failure_threshold(state, failure_threshold=3)
        assert reason_normal is not None
        
        # With anomaly: 5 failures may be < raised threshold
        high_z = _ANOMALY_Z_SCORE_THRESHOLD + 0.1
        reason_anomaly = check_failure_threshold(
            state, failure_threshold=3, signal_z_score=high_z
        )
        
        # Should NOT be inhibited due to raised threshold
        assert reason_anomaly is None
    
    def test_extreme_anomaly_very_high_threshold(self):
        """Extreme anomalies (z>4) should be very hard to inhibit."""
        state = MockState("s1", "engine", consecutive_failures=10)
        
        reason = check_failure_threshold(
            state, failure_threshold=3, signal_z_score=4.0
        )
        
        # Even with 10 failures, extreme anomaly context protects
        assert reason is None
    
    def test_inhibition_during_normal_operation(self):
        """During normal operation (z<2), inhibition works normally."""
        state = MockState("s1", "engine", consecutive_failures=3)
        
        normal_z = 1.5
        reason = check_failure_threshold(
            state, failure_threshold=3, signal_z_score=normal_z
        )
        
        assert reason is not None
    
    def test_evaluate_inhibition_with_context(self):
        """Full inhibition evaluation considers z-score."""
        state = MockState("s1", "engine", consecutive_failures=3)
        
        result = evaluate_inhibition(
            state,
            failure_threshold=3,
            max_hours_without_success=24,
            signal_z_score=3.0,  # High anomaly
        )
        
        # Should NOT be inhibited
        assert not result.is_inhibited


class TestAdaptiveAlpha:
    """R-2: Verify adaptive alpha values per regime."""
    
    def test_regime_alpha_values(self):
        """Alpha values must match R-2 specification."""
        # Import after the values have been updated
        from iot_machine_learning.infrastructure.ml.cognitive.plasticity.base import (
            _REGIME_ALPHA,
        )
        
        assert _REGIME_ALPHA["STABLE"] == 0.10, "STABLE should use conservative alpha"
        assert _REGIME_ALPHA["VOLATILE"] == 0.25, "VOLATILE should use aggressive alpha"
        assert _REGIME_ALPHA["NOISY"] == 0.08, "NOISY should use very conservative alpha"


class TestPrometheusExporter:
    """R-6: Prometheus metrics exporter."""
    
    def setup_method(self):
        """Reset exporter before each test."""
        reset_exporter()
    
    def test_record_latency(self):
        """Latency recording and stats calculation."""
        exporter = PrometheusExporter()
        
        for i in range(10):
            exporter.record_latency("taylor", float(i + 10))
        
        stats = exporter.get_latency_stats("taylor")
        
        assert stats["count"] == 10
        assert stats["avg"] == 14.5
        assert stats["p50"] == 15.0  # Index 5 of sorted [10..19]
    
    def test_record_weight(self):
        """Weight tracking per series."""
        exporter = PrometheusExporter()
        
        exporter.record_weight("sensor_1", "taylor", 0.7)
        exporter.record_weight("sensor_1", "baseline", 0.3)
        exporter.record_weight("sensor_2", "taylor", 0.5)
        
        all_weights = exporter.get_all_weights()
        
        assert all_weights["sensor_1"]["taylor"] == 0.7
        assert all_weights["sensor_1"]["baseline"] == 0.3
        assert all_weights["sensor_2"]["taylor"] == 0.5
    
    def test_anomaly_counter(self):
        """Anomaly override counting."""
        exporter = PrometheusExporter()
        
        exporter.increment_anomaly_override("smart_inhibition")
        exporter.increment_anomaly_override("smart_inhibition")
        exporter.increment_anomaly_override("operator_override")
        
        summary = exporter.get_metrics_summary()
        
        assert summary["anomaly_overrides"] == 3
    
    def test_prometheus_format_output(self):
        """Exported format is Prometheus-compatible."""
        exporter = get_exporter()
        
        exporter.record_latency("taylor", 12.5, "s1")
        exporter.record_weight("s1", "taylor", 0.8)
        exporter.increment_anomaly_override()
        
        output = exporter.export_prometheus_format()
        
        assert "zenin_prediction_latency_ms" in output
        assert "zenin_engine_weight" in output
        assert "zenin_anomaly_overrides_total" in output
        assert 'engine="taylor"' in output
    
    def test_thread_safety(self):
        """Concurrent metric updates are thread-safe."""
        exporter = PrometheusExporter()
        errors: List[Exception] = []
        
        def worker():
            try:
                for i in range(50):
                    exporter.record_latency("taylor", float(i))
                    exporter.record_weight("s1", "taylor", 0.5)
                    exporter.increment_anomaly_override()
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert not errors
        
        summary = exporter.get_metrics_summary()
        assert summary["anomaly_overrides"] == 200  # 4 threads × 50 increments
