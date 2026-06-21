"""
Unit tests for ContextualConfidenceCalculator.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import unittest

from infrastructure.ml.cognitive.explainability.contextual_confidence_calculator import ContextualConfidenceCalculator


class TestContextualConfidenceCalculator(unittest.TestCase):
    """Test cases for ContextualConfidenceCalculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = ContextualConfidenceCalculator()
    
    def test_calculate_confidence_high(self):
        """Test calculating high confidence."""
        confidence = self.calculator.calculate(
            anomaly_score=0.9,
            retrieval_similarity=0.8,
            regime_confidence=0.9,
            feature_stability=0.8,
        )
        
        self.assertGreater(confidence, 0.8)
        self.assertLessEqual(confidence, 1.0)
    
    def test_calculate_confidence_low(self):
        """Test calculating low confidence."""
        confidence = self.calculator.calculate(
            anomaly_score=0.3,
            retrieval_similarity=0.2,
            regime_confidence=0.4,
            feature_stability=0.3,
        )
        
        self.assertLess(confidence, 0.5)
        self.assertGreaterEqual(confidence, 0.0)
    
    def test_calculate_confidence_clamped(self):
        """Test confidence is clamped to [0, 1]."""
        # Test upper bound
        confidence = self.calculator.calculate(
            anomaly_score=1.5,
            retrieval_similarity=1.5,
            regime_confidence=1.5,
            feature_stability=1.5,
        )
        self.assertEqual(confidence, 1.0)
        
        # Test lower bound
        confidence = self.calculator.calculate(
            anomaly_score=-0.5,
            retrieval_similarity=-0.5,
            regime_confidence=-0.5,
            feature_stability=-0.5,
        )
        self.assertEqual(confidence, 0.0)
    
    def test_custom_weights(self):
        """Test custom weights."""
        calculator = ContextualConfidenceCalculator(
            anomaly_weight=0.7,
            retrieval_weight=0.2,
            regime_weight=0.1,
            stability_weight=0.0,
        )
        
        confidence = calculator.calculate(
            anomaly_score=0.8,
            retrieval_similarity=0.5,
            regime_confidence=0.7,
            feature_stability=0.6,
        )
        
        # With custom weights, anomaly score should dominate
        expected = 0.7 * 0.8 + 0.2 * 0.5 + 0.1 * 0.7
        self.assertAlmostEqual(confidence, expected, places=2)


if __name__ == "__main__":
    unittest.main()
