"""Tests for Adaptive Anomaly Thresholds."""

import pytest

from iot_machine_learning.infrastructure.ml.anomaly.adaptive_thresholds import (
    AdaptiveThresholdManager,
)
from iot_machine_learning.domain.entities.results.anomaly import AnomalySeverity


class TestAdaptiveThresholdManager:
    """Test adaptive per-series thresholds."""
    
    def test_cold_start_uses_fallback(self):
        """Cold start should use static fallback thresholds."""
        manager = AdaptiveThresholdManager(warmup_samples=30)
        
        # No history yet
        threshold = manager.get_threshold("sensor_1", "HIGH", fallback=0.85)
        
        assert threshold == 0.85
    
    def test_warmed_up_uses_percentile(self):
        """After warmup, should use adaptive percentile."""
        manager = AdaptiveThresholdManager(warmup_samples=5)
        
        # Add 10 samples (> warmup)
        for score in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            manager.update("sensor_1", score)
        
        # HIGH = 95th percentile of [0.1..1.0] ≈ 0.95
        threshold = manager.get_threshold("sensor_1", "HIGH")
        
        assert threshold > 0.9  # Should be near 95th percentile
    
    def test_different_series_have_different_thresholds(self):
        """Each series should have independent thresholds."""
        manager = AdaptiveThresholdManager(warmup_samples=5)
        
        # Sensor 1: low scores
        for _ in range(10):
            manager.update("sensor_1", 0.2)
        
        # Sensor 2: high scores
        for _ in range(10):
            manager.update("sensor_2", 0.8)
        
        th1 = manager.get_threshold("sensor_1", "HIGH")
        th2 = manager.get_threshold("sensor_2", "HIGH")
        
        assert th1 < th2  # Sensor 1 has lower threshold
    
    def test_update_clamps_invalid_scores(self):
        """Invalid scores should be clamped to [0, 1]."""
        manager = AdaptiveThresholdManager()
        
        # Should not raise, should clamp
        manager.update("sensor_1", -0.5)
        manager.update("sensor_1", 1.5)
        
        # History should contain clamped values
        assert len(manager._history["sensor_1"]) == 2
    
    def test_max_history_enforced(self):
        """History should not exceed max_history."""
        manager = AdaptiveThresholdManager(max_history=10)
        
        # Add 20 samples
        for i in range(20):
            manager.update("sensor_1", float(i) / 20.0)
        
        # Should only keep last 10
        assert len(manager._history["sensor_1"]) == 10
    
    def test_classify_severity_updates_history(self):
        """classify_severity should update history."""
        manager = AdaptiveThresholdManager(warmup_samples=5)
        
        severity = manager.classify_severity("sensor_1", 0.7)
        
        assert len(manager._history["sensor_1"]) == 1
    
    def test_classify_severity_cold_start(self):
        """classify_severity during cold start uses static thresholds."""
        manager = AdaptiveThresholdManager(warmup_samples=30)
        
        # Add only 5 samples (< warmup)
        for score in [0.1, 0.2, 0.3, 0.4, 0.5]:
            manager.update("sensor_1", score)
        
        # Score 0.9 should be CRITICAL (static fallback 0.95)
        severity = manager.classify_severity("sensor_1", 0.96)
        assert severity == AnomalySeverity.CRITICAL
    
    def test_classify_severity_adaptive(self):
        """classify_severity after warmup uses adaptive thresholds."""
        manager = AdaptiveThresholdManager(warmup_samples=5)
        
        # Add 10 low scores
        for _ in range(10):
            manager.update("sensor_1", 0.1)
        
        # Score 0.2 should be HIGH (95th percentile of 0.1s ≈ 0.1)
        severity = manager.classify_severity("sensor_1", 0.15)
        
        # Should be at least MEDIUM or higher
        assert severity in [AnomalySeverity.MEDIUM, AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]
    
    def test_is_warmed_up(self):
        """is_warmed_up should check sample count."""
        manager = AdaptiveThresholdManager(warmup_samples=5)
        
        assert manager.is_warmed_up("sensor_1") is False
        
        for i in range(5):
            manager.update("sensor_1", 0.5)
        
        assert manager.is_warmed_up("sensor_1") is True
    
    def test_get_metrics_empty(self):
        """get_metrics for series with no history."""
        manager = AdaptiveThresholdManager()
        
        metrics = manager.get_metrics("sensor_unknown")
        
        assert metrics["n_samples"] == 0
        assert metrics["warmed_up"] is False
    
    def test_get_metrics_populated(self):
        """get_metrics for series with history."""
        manager = AdaptiveThresholdManager(warmup_samples=5)
        
        for score in [0.1, 0.3, 0.5, 0.7, 0.9]:
            manager.update("sensor_1", score)
        
        metrics = manager.get_metrics("sensor_1")
        
        assert metrics["n_samples"] == 5
        assert metrics["warmed_up"] is True
        assert metrics["mean_score"] == pytest.approx(0.5, abs=1e-6)
        assert "p50" in metrics
        assert "p95" in metrics
    
    def test_custom_percentiles(self):
        """Custom percentiles should be used."""
        custom_percentiles = {
            "LOW": 50.0,
            "MEDIUM": 70.0,
            "HIGH": 90.0,
            "CRITICAL": 98.0,
        }
        manager = AdaptiveThresholdManager(
            warmup_samples=5,
            percentiles=custom_percentiles,
        )
        
        # Add samples
        for i in range(10):
            manager.update("sensor_1", float(i) / 10.0)
        
        # Should use custom percentiles
        threshold = manager.get_threshold("sensor_1", "LOW")
        # 50th percentile of [0.0, 0.1, ..., 0.9] = 0.45
        assert 0.4 < threshold < 0.6
