"""
Unit tests for PropagationConfidenceCalculator.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import unittest

from infrastructure.ml.cognitive.causal.propagation_confidence_calculator import PropagationConfidenceCalculator


class TestPropagationConfidenceCalculator(unittest.TestCase):
    """Test cases for PropagationConfidenceCalculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = PropagationConfidenceCalculator()
    
    def test_calculate_confidence(self):
        """Test calculating propagation confidence."""
        confidence = self.calculator.calculate(
            historical_frequency=0.8,
            temporal_consistency=0.9,
            contextual_stability=0.7,
            operational_correlation=0.85,
        )
        
        self.assertGreater(confidence, 0.7)
        self.assertLessEqual(confidence, 1.0)
    
    def test_calculate_confidence_low(self):
        """Test calculating low confidence."""
        confidence = self.calculator.calculate(
            historical_frequency=0.2,
            temporal_consistency=0.3,
            contextual_stability=0.4,
            operational_correlation=0.5,
        )
        
        self.assertLess(confidence, 0.5)
        self.assertGreaterEqual(confidence, 0.0)
    
    def test_calculate_confidence_clamped(self):
        """Test confidence is clamped to [0, 1]."""
        # Test upper bound
        confidence = self.calculator.calculate(
            historical_frequency=1.5,
            temporal_consistency=1.5,
            contextual_stability=1.5,
            operational_correlation=1.5,
        )
        self.assertEqual(confidence, 1.0)
        
        # Test lower bound
        confidence = self.calculator.calculate(
            historical_frequency=-0.5,
            temporal_consistency=-0.5,
            contextual_stability=-0.5,
            operational_correlation=-0.5,
        )
        self.assertEqual(confidence, 0.0)
    
    def test_calculate_from_statistics(self):
        """Test calculating confidence from statistics."""
        confidence = self.calculator.calculate_from_statistics(
            propagation_count=10,
            total_observations=20,
            duration_variance=10.0,
            context_match_rate=0.8,
            correlation_coefficient=0.85,
        )
        
        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_calculate_from_statistics_no_propagations(self):
        """Test calculating confidence with no propagations."""
        confidence = self.calculator.calculate_from_statistics(
            propagation_count=0,
            total_observations=20,
            duration_variance=10.0,
            context_match_rate=0.8,
            correlation_coefficient=0.85,
        )
        
        self.assertEqual(confidence, 0.0)
    
    def test_calculate_batch(self):
        """Test calculating confidence for multiple statistics."""
        statistics_list = [
            {
                "propagation_count": 10,
                "total_observations": 20,
                "duration_variance": 10.0,
                "context_match_rate": 0.8,
                "correlation_coefficient": 0.85,
            },
            {
                "propagation_count": 5,
                "total_observations": 20,
                "duration_variance": 15.0,
                "context_match_rate": 0.6,
                "correlation_coefficient": 0.75,
            },
        ]
        
        confidences = self.calculator.calculate_batch(statistics_list)
        
        self.assertEqual(len(confidences), 2)
        self.assertTrue(all(0.0 <= c <= 1.0 for c in confidences))
    
    def test_custom_weights(self):
        """Test custom weights."""
        calculator = PropagationConfidenceCalculator(
            frequency_weight=0.5,
            consistency_weight=0.3,
            stability_weight=0.1,
            correlation_weight=0.1,
        )
        
        confidence = calculator.calculate(
            historical_frequency=0.8,
            temporal_consistency=0.9,
            contextual_stability=0.7,
            operational_correlation=0.85,
        )
        
        # With custom weights, frequency should have more impact
        expected = 0.5 * 0.8 + 0.3 * 0.9 + 0.1 * 0.7 + 0.1 * 0.85
        self.assertAlmostEqual(confidence, expected, places=2)


if __name__ == "__main__":
    unittest.main()
