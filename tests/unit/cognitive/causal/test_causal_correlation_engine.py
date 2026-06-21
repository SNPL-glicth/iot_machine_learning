"""
Unit tests for CausalCorrelationEngine.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import unittest

from infrastructure.ml.cognitive.causal.causal_correlation_engine import CausalCorrelationEngine


class TestCausalCorrelationEngine(unittest.TestCase):
    """Test cases for CausalCorrelationEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = CausalCorrelationEngine()
    
    def test_add_sensor_reading(self):
        """Test adding sensor readings."""
        self.engine.add_sensor_reading(sensor_id=12345, value=85.2, timestamp=1234567890.0)
        self.engine.add_sensor_reading(sensor_id=12345, value=86.1, timestamp=1234567891.0)
        
        # Data should be stored
        self.assertEqual(len(self.engine._sensor_data[12345]), 2)
    
    def test_detect_correlations_no_data(self):
        """Test correlation detection with no data."""
        correlations = self.engine.detect_correlations(sensor_id=12345)
        
        self.assertEqual(len(correlations), 0)
    
    def test_detect_correlations_with_data(self):
        """Test correlation detection with data."""
        # Add correlated data
        for i in range(20):
            self.engine.add_sensor_reading(sensor_id=12345, value=50.0 + i, timestamp=1234567890.0 + i)
            self.engine.add_sensor_reading(sensor_id=67890, value=50.0 + i * 0.9, timestamp=1234567890.0 + i + 5)
        
        correlations = self.engine.detect_correlations(sensor_id=12345, target_sensor_ids=[67890])
        
        # Should detect correlation
        self.assertGreater(len(correlations), 0)
    
    def test_compute_lagged_correlation(self):
        """Test lagged correlation computation."""
        # Add data with clear lag
        for i in range(20):
            self.engine.add_sensor_reading(sensor_id=12345, value=50.0 + i, timestamp=1234567890.0 + i)
            self.engine.add_sensor_reading(sensor_id=67890, value=50.0 + i, timestamp=1234567890.0 + i + 10)
        
        source_data = self.engine._sensor_data[12345]
        target_data = self.engine._sensor_data[67890]
        correlation = self.engine._compute_lagged_correlation(12345, 67890, source_data, target_data)
        
        self.assertIsNotNone(correlation)
        self.assertGreater(correlation.correlation_coefficient, 0.5)
        self.assertEqual(correlation.source_sensor_id, 12345)
        self.assertEqual(correlation.target_sensor_id, 67890)
    
    def test_pearson_correlation(self):
        """Test Pearson correlation calculation."""
        from infrastructure.ml.cognitive.causal.utils.correlation_calculator import CorrelationCalculator
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        correlation = CorrelationCalculator.pearson_correlation(x, y)
        
        self.assertAlmostEqual(correlation, 1.0, places=2)
    
    def test_pearson_correlation_negative(self):
        """Test Pearson correlation with negative correlation."""
        from infrastructure.ml.cognitive.causal.utils.correlation_calculator import CorrelationCalculator
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        
        correlation = CorrelationCalculator.pearson_correlation(x, y)
        
        self.assertAlmostEqual(correlation, -1.0, places=2)
    
    def test_calculate_confidence(self):
        """Test confidence calculation."""
        confidence = self.engine._calculate_confidence(0.8, 100)
        
        self.assertGreater(confidence, 0.7)
        self.assertLessEqual(confidence, 1.0)
    
    def test_calculate_propagation_likelihood(self):
        """Test propagation likelihood calculation."""
        likelihood = self.engine._calculate_propagation_likelihood(0.8, 10.0)
        
        self.assertGreater(likelihood, 0.0)
        self.assertLessEqual(likelihood, 1.0)
    
    def test_detect_granger_causality(self):
        """Test Granger causality detection."""
        # Add data
        for i in range(20):
            self.engine.add_sensor_reading(sensor_id=12345, value=50.0 + i, timestamp=1234567890.0 + i)
            self.engine.add_sensor_reading(sensor_id=67890, value=50.0 + i * 0.9, timestamp=1234567890.0 + i + 5)
        
        causality_score = self.engine.detect_granger_causality(12345, 67890)
        
        # May return None if not enough data, or a score
        if causality_score is not None:
            self.assertGreaterEqual(causality_score, 0.0)
            self.assertLessEqual(causality_score, 1.0)
    
    def test_reset(self):
        """Test resetting engine."""
        self.engine.add_sensor_reading(sensor_id=12345, value=85.2, timestamp=1234567890.0)
        
        self.engine.reset()
        
        self.assertEqual(len(self.engine._sensor_data), 0)


if __name__ == "__main__":
    unittest.main()
