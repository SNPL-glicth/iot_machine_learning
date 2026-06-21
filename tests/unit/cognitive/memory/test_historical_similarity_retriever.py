"""
Unit tests for HistoricalSimilarityRetriever.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import unittest
from unittest.mock import Mock

from infrastructure.ml.cognitive.memory.historical_similarity_retriever import HistoricalSimilarityRetriever
from infrastructure.ml.cognitive.memory.anomaly_memory_store import AnomalyMemoryStore
from domain.entities.memory import MemoryEvent


class TestHistoricalSimilarityRetriever(unittest.TestCase):
    """Test cases for HistoricalSimilarityRetriever."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.memory_store = Mock(spec=AnomalyMemoryStore)
        self.retriever = HistoricalSimilarityRetriever(
            memory_store=self.memory_store,
            min_similarity_threshold=0.7,
        )
    
    def test_retrieve_similar_events(self):
        """Test retrieving similar events."""
        ml_features = {
            "current_value": 85.2,
            "baseline": 45.0,
            "z_score": 3.2,
            "dynamic_features": {
                "derivative": 2.5,
                "rolling_std_1h": 8.5,
            },
        }
        
        # Mock memory store response
        mock_event = MemoryEvent(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            timestamp=1234567890.0,
            event_type="ANOMALY_CONFIRMED",
            semantic_text="Sensor 12345 (TEMPERATURE) en régimen STARTUP...",
            regime="STARTUP",
            anomaly_score=0.85,
            dynamic_features={"derivative": 2.5, "rolling_std_1h": 8.5},
            metadata={"value": 85.2, "baseline": 45.0, "z_score": 3.2},
        )
        
        self.memory_store.retrieve_similar.return_value = [mock_event]
        
        events = self.retriever.retrieve(
            sensor_id=12345,
            ml_features=ml_features,
            regime="STARTUP",
            top_k=5,
        )
        
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].sensor_id, 12345)
        self.assertEqual(events[0].regime, "STARTUP")
    
    def test_retrieve_with_filters(self):
        """Test retrieving with filters."""
        ml_features = {
            "current_value": 85.2,
            "baseline": 45.0,
            "z_score": 3.2,
            "dynamic_features": {},
        }
        
        self.memory_store.retrieve_similar.return_value = []
        
        events = self.retriever.retrieve(
            sensor_id=12345,
            ml_features=ml_features,
            regime="STARTUP",
            sensor_type="TEMPERATURE",
            top_k=5,
            time_window=(1234560000.0, 1234567890.0),
        )
        
        self.memory_store.retrieve_similar.assert_called_once()
        call_args = self.memory_store.retrieve_similar.call_args
        self.assertEqual(call_args[1]["sensor_id"], 12345)
        self.assertEqual(call_args[1]["regime"], "STARTUP")
        self.assertEqual(call_args[1]["sensor_type"], "TEMPERATURE")
        self.assertEqual(call_args[1]["time_window"], (1234560000.0, 1234567890.0))
    
    def test_filter_operational_similarity(self):
        """Test filtering by operational similarity."""
        ml_features = {
            "current_value": 85.2,
            "baseline": 45.0,
            "z_score": 3.2,
            "dynamic_features": {},
        }
        
        # Create events with different values
        similar_event = MemoryEvent(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            timestamp=1234567890.0,
            event_type="ANOMALY_CONFIRMED",
            semantic_text="Sensor 12345...",
            regime="STARTUP",
            anomaly_score=0.85,
            dynamic_features={},
            metadata={"value": 86.0, "baseline": 45.0, "z_score": 3.2},
        )
        
        dissimilar_event = MemoryEvent(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            timestamp=1234567890.0,
            event_type="ANOMALY_CONFIRMED",
            semantic_text="Sensor 12345...",
            regime="STARTUP",
            anomaly_score=0.85,
            dynamic_features={},
            metadata={"value": 200.0, "baseline": 45.0, "z_score": 3.2},
        )
        
        events = [similar_event, dissimilar_event]
        
        filtered = self.retriever._filter_operational_similarity(events, ml_features)
        
        # Only similar event should pass (value within 3σ)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].metadata["value"], 86.0)
    
    def test_build_query_text(self):
        """Test building query text."""
        ml_features = {
            "current_value": 85.2,
            "baseline": 45.0,
            "z_score": 3.2,
            "dynamic_features": {
                "derivative": 2.5,
                "rolling_std_1h": 8.5,
            },
        }
        
        query_text = self.retriever._build_query_text(
            sensor_id=12345,
            ml_features=ml_features,
            regime="STARTUP",
        )
        
        self.assertIn("Sensor 12345", query_text)
        self.assertIn("STARTUP", query_text)
        self.assertIn("85.20", query_text)
        self.assertIn("45.00", query_text)
        self.assertIn("3.20", query_text)
        self.assertIn("2.50", query_text)
        self.assertIn("8.50", query_text)


if __name__ == "__main__":
    unittest.main()
