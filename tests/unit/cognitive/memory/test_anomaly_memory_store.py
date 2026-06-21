"""
Unit tests for AnomalyMemoryStore.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import unittest
from unittest.mock import Mock, MagicMock
import time

from infrastructure.ml.cognitive.memory.anomaly_memory_store import AnomalyMemoryStore
from domain.entities.memory import MemoryEvent


class TestAnomalyMemoryStore(unittest.TestCase):
    """Test cases for AnomalyMemoryStore."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.store = AnomalyMemoryStore(
            weaviate_client=self.mock_client,
            embedding_model="text-embedding-3-small",
            batch_size=100,
        )
    
    def test_store_event_success(self):
        """Test storing event successfully."""
        event = MemoryEvent(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            timestamp=time.time(),
            event_type="ANOMALY_CONFIRMED",
            semantic_text="Sensor 12345 (TEMPERATURE) en régimen STARTUP...",
            regime="STARTUP",
            anomaly_score=0.85,
            dynamic_features={"derivative": 2.5, "rolling_std_1h": 8.5},
            metadata={"value": 85.2, "baseline": 45.0, "z_score": 3.2},
        )
        
        self.mock_client.data_object.create.return_value = "test-object-id"
        
        object_id = self.store.store(event, ttl=86400)
        
        self.assertEqual(object_id, "test-object-id")
        self.mock_client.data_object.create.assert_called_once()
    
    def test_store_event_no_client(self):
        """Test storing event when client is not available."""
        self.store._client = None
        
        event = MemoryEvent(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            timestamp=time.time(),
            event_type="ANOMALY_CONFIRMED",
            semantic_text="Sensor 12345...",
            regime="STARTUP",
            anomaly_score=0.85,
            dynamic_features={},
            metadata={},
        )
        
        object_id = self.store.store(event, ttl=86400)
        
        self.assertIsNone(object_id)
    
    def test_store_event_storage_disabled(self):
        """Test storing event when storage is disabled."""
        self.store.enable_storage(False)
        
        event = MemoryEvent(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            timestamp=time.time(),
            event_type="ANOMALY_CONFIRMED",
            semantic_text="Sensor 12345...",
            regime="STARTUP",
            anomaly_score=0.85,
            dynamic_features={},
            metadata={},
        )
        
        object_id = self.store.store(event, ttl=86400)
        
        self.assertIsNone(object_id)
    
    def test_retrieve_similar_events(self):
        """Test retrieving similar events."""
        query_embedding = [0.1] * 1536
        
        # Mock Weaviate response
        mock_result = {
            "properties": {
                "sensor_id": 12345,
                "sensor_type": "TEMPERATURE",
                "timestamp": 1234567890.0,
                "event_type": "ANOMALY_CONFIRMED",
                "semantic_text": "Sensor 12345...",
                "regime": "STARTUP",
                "anomaly_score": 0.85,
                "dynamic_features": {"derivative": 2.5},
                "metadata": {"value": 85.2, "baseline": 45.0, "z_score": 3.2},
            }
        }
        
        self.mock_client.query.get.return_value = [mock_result]
        
        events = self.store.retrieve_similar(
            query_embedding=query_embedding,
            sensor_id=12345,
            regime="STARTUP",
            top_k=5,
        )
        
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].sensor_id, 12345)
        self.assertEqual(events[0].regime, "STARTUP")
    
    def test_retrieve_similar_no_client(self):
        """Test retrieving when client is not available."""
        self.store._client = None
        
        events = self.store.retrieve_similar(
            query_embedding=[0.1] * 1536,
            sensor_id=12345,
            regime="STARTUP",
            top_k=5,
        )
        
        self.assertEqual(len(events), 0)
    
    def test_cleanup_expired(self):
        """Test cleaning up expired events."""
        # Mock expired events
        expired_events = [
            {"id": "expired-1"},
            {"id": "expired-2"},
        ]
        
        self.mock_client.query.get.return_value = expired_events
        
        count = self.store.cleanup_expired()
        
        self.assertEqual(count, 2)
        self.assertEqual(self.mock_client.data_object.delete.call_count, 2)
    
    def test_cleanup_expired_no_client(self):
        """Test cleanup when client is not available."""
        self.store._client = None
        
        count = self.store.cleanup_expired()
        
        self.assertEqual(count, 0)
    
    def test_generate_embedding(self):
        """Test embedding generation."""
        text = "Sensor 12345 (TEMPERATURE) en régimen STARTUP"
        
        embedding = self.store._generate_embedding(text)
        
        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 1536)  # OpenAI default dimension
        self.assertTrue(all(isinstance(x, float) for x in embedding))
    
    def test_build_filter(self):
        """Test building Weaviate filter."""
        filter_obj = self.store._build_filter(
            sensor_id=12345,
            regime="STARTUP",
            sensor_type="TEMPERATURE",
            time_window=(1234560000.0, 1234567890.0),
        )
        
        self.assertIsNotNone(filter_obj)
        self.assertEqual(filter_obj["operator"], "And")
        self.assertEqual(len(filter_obj["operands"]), 4)
    
    def test_build_filter_no_filters(self):
        """Test building filter with no filters."""
        filter_obj = self.store._build_filter(
            sensor_id=None,
            regime=None,
            sensor_type=None,
            time_window=None,
        )
        
        self.assertIsNone(filter_obj)
    
    def test_result_to_event(self):
        """Test converting Weaviate result to MemoryEvent."""
        result = {
            "properties": {
                "sensor_id": 12345,
                "sensor_type": "TEMPERATURE",
                "timestamp": 1234567890.0,
                "event_type": "ANOMALY_CONFIRMED",
                "semantic_text": "Sensor 12345...",
                "regime": "STARTUP",
                "anomaly_score": 0.85,
                "dynamic_features": {"derivative": 2.5},
                "metadata": {"value": 85.2, "baseline": 45.0, "z_score": 3.2},
            }
        }
        
        event = self.store._result_to_event(result)
        
        self.assertIsInstance(event, MemoryEvent)
        self.assertEqual(event.sensor_id, 12345)
        self.assertEqual(event.sensor_type, "TEMPERATURE")
        self.assertEqual(event.regime, "STARTUP")


if __name__ == "__main__":
    unittest.main()
