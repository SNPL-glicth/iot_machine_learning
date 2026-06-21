"""
Integration tests for contextual explainability with operational memory.

These tests require operational memory to be available.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import unittest
from unittest.mock import Mock

from infrastructure.ml.cognitive.explainability.contextual_explainability_engine import ContextualExplainabilityEngine
from infrastructure.ml.cognitive.memory.historical_similarity_retriever import HistoricalSimilarityRetriever
from infrastructure.ml.cognitive.memory.anomaly_memory_store import AnomalyMemoryStore
from infrastructure.ml.cognitive.memory.cognitive_memory_registry import CognitiveMemoryRegistry
from domain.entities.memory import MemoryEvent


class TestExplainabilityIntegration(unittest.TestCase):
    """Integration tests for contextual explainability."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_store = Mock(spec=AnomalyMemoryStore)
        self.mock_store.retrieve_similar.return_value = []
        
        self.retriever = HistoricalSimilarityRetriever(
            memory_store=self.mock_store,
            min_similarity_threshold=0.7,
        )
        
        self.registry = CognitiveMemoryRegistry()
        
        self.engine = ContextualExplainabilityEngine(
            similarity_retriever=self.retriever,
            registry=self.registry,
        )
    
    def test_explainability_with_memory_retrieval(self):
        """Test explainability with memory retrieval."""
        # Mock similar events
        similar_event = MemoryEvent(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            timestamp=1234567890.0,
            event_type="ANOMALY_CONFIRMED",
            semantic_text="Sensor 12345...",
            regime="STARTUP",
            anomaly_score=0.85,
            dynamic_features={"derivative": 2.5},
            metadata={"value": 85.2},
        )
        
        self.mock_store.retrieve_similar.return_value = [similar_event]
        
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
        
        self.assertEqual(explanation.similar_event_count, 1)
        self.assertIn("1 evento similar", explanation.historical_context)
    
    def test_explainability_without_memory_retrieval(self):
        """Test explainability without memory retrieval."""
        self.registry.enable_retrieval_feature(False)
        
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
        
        self.assertEqual(explanation.similar_event_count, 0)
        self.mock_store.retrieve_similar.assert_not_called()
    
    def test_explainability_consistency(self):
        """Test explainability consistency across multiple calls."""
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
        
        # Generate explanation twice
        explanation1 = self.engine.generate_explanation(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            ml_features=ml_features,
            regime="STARTUP",
            anomaly_score=0.85,
            regime_confidence=0.9,
        )
        
        explanation2 = self.engine.generate_explanation(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            ml_features=ml_features,
            regime="STARTUP",
            anomaly_score=0.85,
            regime_confidence=0.9,
        )
        
        # Should be consistent
        self.assertEqual(explanation1.sensor_id, explanation2.sensor_id)
        self.assertEqual(explanation1.current_regime, explanation2.current_regime)
        self.assertEqual(explanation1.anomaly_score, explanation2.anomaly_score)
        self.assertEqual(explanation1.primary_drivers, explanation2.primary_drivers)


if __name__ == "__main__":
    unittest.main()
