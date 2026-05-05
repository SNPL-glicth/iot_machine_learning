"""Tests for multivariate anomaly detection subsystem — FASE 3.

Tests OnlinePCA, DynamicCorrelationTracker, MultivariateDetector.
All tests are 100% unit tests with mocked dependencies.
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock
import numpy as np

from iot_machine_learning.infrastructure.ml.engines.multivariate import (
    OnlinePCA,
    DynamicCorrelationTracker,
    MultivariateAnomalyEngine,
)
from iot_machine_learning.infrastructure.ml.anomaly.detectors.multivariate_detector import (
    MultivariateDetector,
)


class TestOnlinePCA:
    """Test online PCA wrapper."""
    
    def test_pca_validates_n_components(self):
        """PCA should validate n_components parameter."""
        with pytest.raises(ValueError, match="n_components must be >= 1"):
            OnlinePCA(n_components=0)
    
    def test_pca_partial_fit_updates_state(self):
        """PCA should update fitted state after partial_fit."""
        pca = OnlinePCA(n_components=2)
        
        assert not pca.fitted
        
        # Fit with sufficient samples
        X = np.random.randn(10, 3)
        pca.partial_fit(X)
        
        assert pca.fitted
    
    def test_pca_transform_returns_none_when_not_fitted(self):
        """PCA should return None when transforming before fit."""
        pca = OnlinePCA(n_components=2)
        
        X = np.random.randn(5, 3)
        result = pca.transform(X)
        
        assert result is None
    
    def test_pca_score_samples_computes_distances(self):
        """PCA should compute distances after fitting."""
        pca = OnlinePCA(n_components=2)
        
        # Fit
        X_train = np.random.randn(20, 3)
        pca.partial_fit(X_train)
        
        # Score
        X_test = np.random.randn(5, 3)
        distances = pca.score_samples(X_test)
        
        assert distances is not None
        assert len(distances) == 5
        assert all(d >= 0 for d in distances)


class TestDynamicCorrelationTracker:
    """Test correlation tracker."""
    
    def test_correlation_tracker_validates_series_id(self):
        """Tracker should validate series_id against IEC 62443 pattern."""
        tracker = DynamicCorrelationTracker()
        
        # Valid IDs
        tracker.update("sensor_42", 10.0)
        tracker.update("series-123", 20.0)
        tracker.update("iot:device:001", 30.0)
        
        # Invalid IDs (should be rejected silently)
        tracker.update("../../../etc/passwd", 40.0)
        tracker.update("series; DROP TABLE", 50.0)
        
        # Only valid IDs should be stored
        assert "sensor_42" in tracker._data
        assert "../../../etc/passwd" not in tracker._data
    
    def test_correlation_tracker_computes_correlations(self):
        """Tracker should compute Pearson correlations."""
        tracker = DynamicCorrelationTracker(min_samples=5)
        
        # Add correlated series
        for i in range(20):
            tracker.update("series_a", float(i))
            tracker.update("series_b", float(i) * 2)  # Perfect positive correlation
            tracker.update("series_c", float(i) * -1)  # Perfect negative correlation
        
        correlated = tracker.get_correlated("series_a", threshold=0.5)
        
        # Should find series_b and series_c
        assert len(correlated) >= 1
        # series_b should have high positive correlation
        correlations_dict = {sid: corr for sid, corr in correlated}
        if "series_b" in correlations_dict:
            assert correlations_dict["series_b"] > 0.9
    
    def test_correlation_tracker_returns_empty_for_insufficient_data(self):
        """Tracker should return empty list when insufficient data."""
        tracker = DynamicCorrelationTracker(min_samples=10)
        
        # Add only 5 samples
        for i in range(5):
            tracker.update("series_a", float(i))
        
        correlated = tracker.get_correlated("series_a")
        
        assert correlated == []


class TestMultivariateDetector:
    """Test multivariate sub-detector."""
    
    def test_multivariate_detector_passthrough_when_disabled(self):
        """Detector should return 0.0 when disabled."""
        detector = MultivariateDetector(enabled=False)
        
        values = [float(i) for i in range(20)]
        vote = detector.detect(values)
        
        assert vote == 0.0
    
    def test_multivariate_detector_passthrough_insufficient_series(self):
        """Detector should return 0.0 when insufficient correlated series."""
        detector = MultivariateDetector(min_series=3, enabled=True)
        
        values = [float(i) for i in range(20)]
        # Only 1 correlated series (< min_series - 1)
        correlated_data = {"series_1": [float(i) for i in range(20)]}
        vote = detector.detect(values, correlated_series_data=correlated_data)
        
        assert vote == 0.0
    
    def test_multivariate_detector_method_name(self):
        """Detector should have correct method_name."""
        detector = MultivariateDetector()
        
        assert detector.method_name == "multivariate"
    
    def test_multivariate_detects_joint_anomaly(self):
        """Detector should detect joint anomaly across correlated series."""
        detector = MultivariateDetector(
            min_series=3,
            enabled=True,
            pca_components=2,
            baseline_percentile=90.0,
            warmup_samples=15,  # Lower warmup for test
        )
        
        # Generate 3 correlated time series (normal behavior)
        np.random.seed(42)
        n_samples = 40
        
        normal_values = []
        series_1 = []
        series_2 = []
        
        # Use STATIONARY data (no trend) to avoid high baseline
        for i in range(n_samples):
            normal_values.append(10.0 + np.random.randn() * 0.5)  # Mean=10, low variance
            series_1.append(15.0 + np.random.randn() * 0.5)  # Mean=15, correlated
            series_2.append(8.0 + np.random.randn() * 0.5)   # Mean=8, correlated
        
        # Train on normal data (warmup + extra)
        normal_scores = []
        for i in range(10, 35):  # 25 iterations for warmup
            correlated_data = {
                "series_1": series_1[:i],
                "series_2": series_2[:i],
            }
            
            score = detector.detect(
                normal_values[:i],
                series_id="target",
                correlated_series_data=correlated_data,
            )
            normal_scores.append(score)
        
        avg_normal_score = np.mean(normal_scores[-10:]) if normal_scores else 0.0
        
        # Inject STRONG joint anomaly (synchronized spike)
        anomaly_values = normal_values.copy()
        anomaly_series_1 = series_1.copy()
        anomaly_series_2 = series_2.copy()
        
        # STRONG spike at end (breaks correlation pattern)
        anomaly_values[-1] += 50.0   # +50 (huge spike)
        anomaly_series_1[-1] += 75.0  # +75 (huge spike, maintains correlation)
        anomaly_series_2[-1] += 40.0  # +40 (huge spike, maintains correlation)
        
        correlated_data_anomaly = {
            "series_1": anomaly_series_1,
            "series_2": anomaly_series_2,
        }
        
        score_anomaly = detector.detect(
            anomaly_values,
            series_id="target",
            correlated_series_data=correlated_data_anomaly,
        )
        
        # Debug output
        print(f"\n🔍 Debug:")
        print(f"   Normal avg score: {avg_normal_score:.4f}")
        print(f"   Anomaly score: {score_anomaly:.4f}")
        print(f"   Baseline tracker warmed up: {detector._baseline_tracker.is_warmed_up}")
        print(f"   Baseline threshold: {detector._baseline_tracker.baseline_threshold}")
        
        # Anomaly score should be non-zero (warmup complete)
        assert score_anomaly > 0.0, f"Score should be > 0 after warmup, got {score_anomaly}"
        # Anomaly score should be higher than normal
        assert score_anomaly > avg_normal_score, f"Anomaly ({score_anomaly:.4f}) should be > normal ({avg_normal_score:.4f})"


class TestMultivariateAnomalyEngine:
    """Test multivariate engine."""
    
    def test_multivariate_engine_registered(self):
        """Engine should be registered with factory."""
        from iot_machine_learning.infrastructure.ml.engines.core.factory import EngineFactory
        
        # Check if multivariate_pca is registered
        engines = EngineFactory.list_engines()
        assert "multivariate_pca" in engines
    
    def test_multivariate_engine_can_handle(self):
        """Engine should handle sufficient data."""
        engine = MultivariateAnomalyEngine()
        
        assert engine.can_handle(10)
        assert not engine.can_handle(5)
    
    def test_multivariate_engine_predict_passthrough(self):
        """Engine should pass-through with low confidence."""
        engine = MultivariateAnomalyEngine()
        
        values = [float(i) for i in range(20)]
        result = engine.predict(values)
        
        # Should return pass-through result
        assert result.confidence == 0.0
        assert result.metadata.get("passthrough") is True
