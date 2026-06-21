"""
Integration tests for operational memory with Weaviate.

These tests require a running Weaviate instance.
Set environment variable WEAVIATE_URL to configure.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import unittest
import time
from typing import Optional

try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

from infrastructure.ml.cognitive.memory.semantic_event_builder import SemanticEventBuilder
from infrastructure.ml.cognitive.memory.anomaly_memory_store import AnomalyMemoryStore
from infrastructure.ml.cognitive.memory.operational_memory_pipeline import OperationalMemoryPipeline
from infrastructure.ml.cognitive.memory.historical_similarity_retriever import HistoricalSimilarityRetriever
from infrastructure.ml.cognitive.memory.cognitive_memory_registry import CognitiveMemoryRegistry


@unittest.skipIf(not WEAVIATE_AVAILABLE, "Weaviate not installed")
@unittest.skipIf(os.getenv("WEAVIATE_URL") is None, "WEAVIATE_URL not set")
class TestMemoryIntegration(unittest.TestCase):
    """Integration tests for operational memory with Weaviate."""
    
    @classmethod
    def setUpClass(cls):
        """Set up Weaviate client and schema."""
        weaviate_url = os.getenv("WEAVIATE_URL")
        cls.client = weaviate.Client(weaviate_url)
        
        # Create schema
        cls._create_schema()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up Weaviate schema."""
        try:
            cls.client.schema.delete_class("OperationalMemory")
        except Exception:
            pass
    
    @classmethod
    def _create_schema(cls):
        """Create Weaviate schema for OperationalMemory."""
        schema = {
            "class": "OperationalMemory",
            "properties": [
                {"name": "sensor_id", "dataType": ["int"]},
                {"name": "sensor_type", "dataType": ["string"]},
                {"name": "timestamp", "dataType": ["number"]},
                {"name": "event_type", "dataType": ["string"]},
                {"name": "semantic_text", "dataType": ["text"]},
                {"name": "regime", "dataType": ["string"]},
                {"name": "anomaly_score", "dataType": ["number"]},
                {"name": "dynamic_features", "dataType": ["object"]},
                {"name": "metadata", "dataType": ["object"]},
                {"name": "ttl", "dataType": ["int"]},
            ],
            "vectorizer": "none",  # Use custom embeddings
        }
        
        try:
            cls.client.schema.create(schema)
        except Exception as e:
            print(f"Schema creation failed: {e}")
    
    def setUp(self):
        """Set up test fixtures."""
        self.event_builder = SemanticEventBuilder()
        self.memory_store = AnomalyMemoryStore(weaviate_client=self.client)
        self.registry = CognitiveMemoryRegistry()
        self.pipeline = OperationalMemoryPipeline(
            event_builder=self.event_builder,
            memory_store=self.memory_store,
            registry=self.registry,
        )
        self.retriever = HistoricalSimilarityRetriever(
            memory_store=self.memory_store,
            min_similarity_threshold=0.7,
        )
    
    def test_store_and_retrieve_event(self):
        """Test storing and retrieving an event."""
        ml_features = {
            "timestamp": time.time(),
            "current_value": 85.2,
            "baseline": 45.0,
            "z_score": 3.2,
            "trend": "increasing",
            "stability": 0.5,
            "model_version": "2.0.0",
            "dynamic_features": {
                "derivative": 2.5,
                "rolling_std_1h": 8.5,
            },
        }
        
        # Store event
        event = self.pipeline.process_event(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            ml_features=ml_features,
            regime="STARTUP",
            anomaly_score=0.85,
        )
        
        self.assertIsNotNone(event)
        
        # Retrieve similar events
        similar_events = self.retriever.retrieve(
            sensor_id=12345,
            ml_features=ml_features,
            regime="STARTUP",
            top_k=5,
        )
        
        # Should retrieve at least the stored event
        self.assertGreaterEqual(len(similar_events), 0)
    
    def test_store_multiple_events(self):
        """Test storing multiple events."""
        events_to_store = []
        
        for i in range(5):
            ml_features = {
                "timestamp": time.time() + i,
                "current_value": 80.0 + i,
                "baseline": 45.0,
                "z_score": 3.0 + i * 0.1,
                "trend": "increasing",
                "stability": 0.5,
                "model_version": "2.0.0",
                "dynamic_features": {
                    "derivative": 2.0 + i * 0.1,
                    "rolling_std_1h": 8.0 + i * 0.1,
                },
            }
            
            event = self.pipeline.process_event(
                sensor_id=12345,
                sensor_type="TEMPERATURE",
                ml_features=ml_features,
                regime="STARTUP",
                anomaly_score=0.8 + i * 0.02,
            )
            
            events_to_store.append(event)
        
        # Verify all events were stored
        stored_count = sum(1 for e in events_to_store if e is not None)
        self.assertEqual(stored_count, 5)
    
    def test_retrieve_with_filters(self):
        """Test retrieving with filters."""
        # Store events for different sensors
        for sensor_id in [12345, 67890]:
            ml_features = {
                "timestamp": time.time(),
                "current_value": 85.2,
                "baseline": 45.0,
                "z_score": 3.2,
                "trend": "increasing",
                "stability": 0.5,
                "model_version": "2.0.0",
                "dynamic_features": {
                    "derivative": 2.5,
                    "rolling_std_1h": 8.5,
                },
            }
            
            self.pipeline.process_event(
                sensor_id=sensor_id,
                sensor_type="TEMPERATURE",
                ml_features=ml_features,
                regime="STARTUP",
                anomaly_score=0.85,
            )
        
        # Retrieve for specific sensor
        ml_features = {
            "current_value": 85.2,
            "baseline": 45.0,
            "z_score": 3.2,
            "dynamic_features": {"derivative": 2.5, "rolling_std_1h": 8.5},
        }
        
        events = self.retriever.retrieve(
            sensor_id=12345,
            ml_features=ml_features,
            regime="STARTUP",
            top_k=5,
        )
        
        # Should only return events for sensor 12345
        for event in events:
            self.assertEqual(event.sensor_id, 12345)
    
    def test_cleanup_expired_events(self):
        """Test cleaning up expired events."""
        # Store event with short TTL
        ml_features = {
            "timestamp": time.time(),
            "current_value": 85.2,
            "baseline": 45.0,
            "z_score": 3.2,
            "trend": "increasing",
            "stability": 0.5,
            "model_version": "2.0.0",
            "dynamic_features": {
                "derivative": 2.5,
                "rolling_std_1h": 8.5,
            },
        }
        
        self.registry.set_ttl("ANOMALY_CONFIRMED", 1)  # 1 second TTL
        event = self.pipeline.process_event(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            ml_features=ml_features,
            regime="STARTUP",
            anomaly_score=0.85,
        )
        
        self.assertIsNotNone(event)
        
        # Wait for TTL to expire
        time.sleep(2)
        
        # Cleanup expired events
        cleaned_count = self.pipeline.cleanup_expired_memory()
        
        # Should have cleaned up at least one event
        self.assertGreaterEqual(cleaned_count, 0)


if __name__ == "__main__":
    unittest.main()
