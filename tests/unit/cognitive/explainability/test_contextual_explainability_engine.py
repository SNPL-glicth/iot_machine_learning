"""
Unit tests for ContextualExplainabilityEngine.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import unittest
from unittest.mock import Mock

from infrastructure.ml.cognitive.explainability.contextual_explainability_engine import ContextualExplainabilityEngine
from infrastructure.ml.cognitive.memory.cognitive_memory_registry import CognitiveMemoryRegistry
from domain.entities.explainability import ContextualExplanation


class TestContextualExplainabilityEngine(unittest.TestCase):
    """Test cases for ContextualExplainabilityEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_retriever = Mock()
        self.mock_retriever.retrieve.return_value = []
        self.registry = CognitiveMemoryRegistry()
        
        self.engine = ContextualExplainabilityEngine(
            similarity_retriever=self.mock_retriever,
            registry=self.registry,
        )
    
    def test_generate_explanation_basic(self):
        """Test generating basic explanation."""
        ml_features = {
            "timestamp": 1234567890.0,
            "current_value": 85.2,
            "baseline": 45.0,
            "z_score": 3.2,
            "trend": "increasing",
            "stability": 0.5,
            "dynamic_features": {
                "derivative": 2.5,
                "rolling_std_1h": 8.5,
            },
        }
        
        explanation = self.engine.generate_explanation(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            ml_features=ml_features,
            regime="STARTUP",
            anomaly_score=0.85,
            regime_confidence=0.9,
        )
        
        self.assertIsInstance(explanation, ContextualExplanation)
        self.assertEqual(explanation.sensor_id, 12345)
        self.assertEqual(explanation.sensor_type, "TEMPERATURE")
        self.assertEqual(explanation.current_regime, "STARTUP")
        self.assertEqual(explanation.anomaly_score, 0.85)
    
    def test_generate_explanation_with_retrieval_disabled(self):
        """Test generating explanation with retrieval disabled."""
        self.registry.enable_retrieval_feature(False)
        
        ml_features = {
            "timestamp": 1234567890.0,
            "current_value": 85.2,
            "baseline": 45.0,
            "z_score": 3.2,
            "trend": "increasing",
            "stability": 0.5,
            "dynamic_features": {},
        }
        
        explanation = self.engine.generate_explanation(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            ml_features=ml_features,
            regime="STARTUP",
            anomaly_score=0.85,
        )
        
        self.assertEqual(explanation.similar_event_count, 0)
        self.mock_retriever.retrieve.assert_not_called()
    
    def test_identify_primary_drivers(self):
        """Test identifying primary drivers."""
        ml_features = {
            "z_score": 3.2,
            "dynamic_features": {
                "derivative": 2.5,
                "rolling_std_1h": 8.5,
            },
        }
        
        drivers = self.engine._identify_primary_drivers(ml_features, anomaly_score=0.85)
        
        self.assertIsInstance(drivers, list)
        self.assertGreater(len(drivers), 0)
        self.assertTrue(any("Z-score" in d for d in drivers))
    
    def test_extract_dynamic_context(self):
        """Test extracting dynamic context."""
        ml_features = {
            "current_value": 85.2,
            "baseline": 45.0,
            "z_score": 3.2,
            "trend": "increasing",
            "stability": 0.5,
            "dynamic_features": {
                "derivative": 2.5,
                "rolling_std_1h": 8.5,
            },
        }
        
        context = self.engine._extract_dynamic_context(ml_features)
        
        self.assertIn("current_value", context)
        self.assertIn("baseline", context)
        self.assertIn("z_score", context)
        self.assertIn("derivative", context)
        self.assertEqual(context["current_value"], 85.2)
    
    def test_operational_confidence_calculation(self):
        """Test operational confidence calculation."""
        ml_features = {
            "timestamp": 1234567890.0,
            "current_value": 85.2,
            "baseline": 45.0,
            "z_score": 3.2,
            "trend": "increasing",
            "stability": 0.8,
            "dynamic_features": {},
        }
        
        explanation = self.engine.generate_explanation(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            ml_features=ml_features,
            regime="STARTUP",
            anomaly_score=0.85,
            regime_confidence=0.9,
        )
        
        self.assertGreater(explanation.operational_confidence, 0.0)
        self.assertLessEqual(explanation.operational_confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
