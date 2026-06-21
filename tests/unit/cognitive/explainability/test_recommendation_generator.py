"""
Unit tests for RecommendationGenerator.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import unittest

from infrastructure.ml.cognitive.explainability.recommendation_generator import RecommendationGenerator


class TestRecommendationGenerator(unittest.TestCase):
    """Test cases for RecommendationGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = RecommendationGenerator()
    
    def test_generate_startup_recommendations(self):
        """Test generating recommendations for STARTUP regime."""
        recommendations = self.generator.generate(
            regime="STARTUP",
            anomaly_score=0.7,
            dynamic_features={"derivative": 1.5, "rolling_std_1h": 3.0},
            historical_patterns=["STARTUP"],
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        self.assertTrue(any("ramp-up" in r.lower() for r in recommendations))
    
    def test_generate_shutdown_recommendations(self):
        """Test generating recommendations for SHUTDOWN regime."""
        recommendations = self.generator.generate(
            regime="SHUTDOWN",
            anomaly_score=0.5,
            dynamic_features={},
            historical_patterns=[],
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        self.assertTrue(any("cooldown" in r.lower() or "apagado" in r.lower() for r in recommendations))
    
    def test_generate_volatile_peak_recommendations(self):
        """Test generating recommendations for VOLATILE_PEAK regime."""
        recommendations = self.generator.generate(
            regime="VOLATILE_PEAK",
            anomaly_score=0.8,
            dynamic_features={"rolling_std_1h": 6.0},
            historical_patterns=["VOLATILE_PEAK"],
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        self.assertTrue(any("carga" in r.lower() for r in recommendations))
    
    def test_generate_high_anomaly_recommendations(self):
        """Test generating recommendations for high anomaly score."""
        recommendations = self.generator.generate(
            regime="STABLE_NORMAL",
            anomaly_score=0.9,
            dynamic_features={},
            historical_patterns=[],
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        self.assertTrue(any("alta prioridad" in r.lower() for r in recommendations))
    
    def test_generate_derivative_recommendations(self):
        """Test generating recommendations for high derivative."""
        recommendations = self.generator.generate(
            regime="STABLE_NORMAL",
            anomaly_score=0.7,
            dynamic_features={"derivative": 3.0},
            historical_patterns=[],
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertTrue(any("tasa de cambio" in r.lower() for r in recommendations))
    
    def test_generate_volatility_recommendations(self):
        """Test generating recommendations for high volatility."""
        recommendations = self.generator.generate(
            regime="STABLE_NORMAL",
            anomaly_score=0.6,
            dynamic_features={"rolling_std_1h": 7.0},
            historical_patterns=[],
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertTrue(any("volatilidad" in r.lower() for r in recommendations))
    
    def test_generate_historical_pattern_recommendations(self):
        """Test generating recommendations based on historical patterns."""
        recommendations = self.generator.generate(
            regime="STARTUP",
            anomaly_score=0.7,
            dynamic_features={},
            historical_patterns=["STARTUP", "VOLATILE_PEAK"],
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertTrue(any("startup" in r.lower() for r in recommendations))


if __name__ == "__main__":
    unittest.main()
