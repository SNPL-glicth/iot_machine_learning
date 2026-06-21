"""
Unit tests for DriftDetectionEngine.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import unittest

from infrastructure.ml.cognitive.observability.drift_detection_engine import DriftDetectionEngine


class TestDriftDetectionEngine(unittest.TestCase):
    """Test cases for DriftDetectionEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = DriftDetectionEngine()
        
        # Set baselines
        self.engine.set_baselines(
            regime_distribution={"STARTUP": 0.3, "STABLE_NORMAL": 0.7},
            feature_means={"temperature": 50.0, "pressure": 100.0},
            anomaly_frequency=0.1,
            embedding_mean=0.5,
        )
    
    def test_detect_no_drift(self):
        """Test detecting no drift."""
        current_regime = {"STARTUP": 0.3, "STABLE_NORMAL": 0.7}
        current_features = {"temperature": 50.0, "pressure": 100.0}
        current_anomaly = 0.1
        current_embedding = 0.5
        
        result = self.engine.detect_drift(
            current_regime_distribution=current_regime,
            current_feature_means=current_features,
            current_anomaly_frequency=current_anomaly,
            current_embedding_mean=current_embedding,
        )
        
        self.assertFalse(result.drift_detected)
        self.assertEqual(result.drift_magnitude, 0.0)
    
    def test_detect_statistical_drift(self):
        """Test detecting statistical drift."""
        current_features = {"temperature": 75.0, "pressure": 100.0}  # 50% increase in temperature
        
        result = self.engine.detect_drift(
            current_regime_distribution={"STARTUP": 0.3, "STABLE_NORMAL": 0.7},
            current_feature_means=current_features,
            current_anomaly_frequency=0.1,
            current_embedding_mean=0.5,
        )
        
        self.assertTrue(result.drift_detected)
        self.assertEqual(result.drift_type, "statistical")
        self.assertGreater(result.drift_magnitude, 0.3)
    
    def test_detect_regime_drift(self):
        """Test detecting regime drift."""
        current_regime = {"STARTUP": 0.7, "STABLE_NORMAL": 0.3}  # Regime shift
        
        result = self.engine.detect_drift(
            current_regime_distribution=current_regime,
            current_feature_means={"temperature": 50.0, "pressure": 100.0},
            current_anomaly_frequency=0.1,
            current_embedding_mean=0.5,
        )
        
        self.assertTrue(result.drift_detected)
        self.assertEqual(result.drift_type, "regime")
    
    def test_detect_anomaly_frequency_drift(self):
        """Test detecting anomaly frequency drift."""
        current_anomaly = 0.2  # 100% increase in anomaly frequency
        
        result = self.engine.detect_drift(
            current_regime_distribution={"STARTUP": 0.3, "STABLE_NORMAL": 0.7},
            current_feature_means={"temperature": 50.0, "pressure": 100.0},
            current_anomaly_frequency=current_anomaly,
            current_embedding_mean=0.5,
        )
        
        self.assertTrue(result.drift_detected)
        self.assertEqual(result.drift_type, "anomaly_frequency")
    
    def test_detect_with_sensor_and_regime(self):
        """Test drift detection with sensor and regime context."""
        result = self.engine.detect_drift(
            current_regime_distribution={"STARTUP": 0.7, "STABLE_NORMAL": 0.3},
            current_feature_means={"temperature": 50.0, "pressure": 100.0},
            current_anomaly_frequency=0.1,
            current_embedding_mean=0.5,
            sensor_id=12345,
            regime="STARTUP",
            temporal_window=(1234560000.0, 1234567890.0),
        )
        
        self.assertEqual(result.drift_sensor_id, 12345)
        self.assertEqual(result.drift_regime, "STARTUP")
        self.assertEqual(result.drift_temporal_window, (1234560000.0, 1234567890.0))
    
    def test_determine_drift_type_and_magnitude(self):
        """Test determining drift type and magnitude."""
        drift_type, magnitude = self.engine._determine_drift_type_and_magnitude(
            statistical_drift=0.5,
            regime_drift=0.3,
            feature_drift=0.2,
            anomaly_drift=0.1,
            embedding_drift=0.05,
        )
        
        self.assertEqual(drift_type, "statistical")
        self.assertEqual(magnitude, 0.5)


if __name__ == "__main__":
    unittest.main()
