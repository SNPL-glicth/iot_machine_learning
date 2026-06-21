"""
Unit tests for OperationalMemoryPipeline.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import unittest
from unittest.mock import Mock, MagicMock
import time

from infrastructure.ml.cognitive.memory.operational_memory_pipeline import OperationalMemoryPipeline
from infrastructure.ml.cognitive.memory.semantic_event_builder import SemanticEventBuilder
from infrastructure.ml.cognitive.memory.anomaly_memory_store import AnomalyMemoryStore
from infrastructure.ml.cognitive.memory.cognitive_memory_registry import CognitiveMemoryRegistry


class TestOperationalMemoryPipeline(unittest.TestCase):
    """Test cases for OperationalMemoryPipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.event_builder = SemanticEventBuilder()
        self.memory_store = Mock(spec=AnomalyMemoryStore)
        self.registry = CognitiveMemoryRegistry()
        
        self.pipeline = OperationalMemoryPipeline(
            event_builder=self.event_builder,
            memory_store=self.memory_store,
            registry=self.registry,
        )
    
    def test_process_event_anomaly_confirmed(self):
        """Test processing an anomaly confirmed event."""
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
        
        event = self.pipeline.process_event(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            ml_features=ml_features,
            regime="STARTUP",
            anomaly_score=0.85,
        )
        
        self.assertIsNotNone(event)
        self.assertEqual(event.sensor_id, 12345)
        self.assertEqual(event.event_type, "ANOMALY_CONFIRMED")
    
    def test_process_event_low_quality(self):
        """Test processing low quality event (should be filtered)."""
        ml_features = {
            "timestamp": time.time(),
            "current_value": 45.0,
            "baseline": 44.0,
            "z_score": 0.3,
            "trend": "stable",
            "stability": 0.95,
            "model_version": "2.0.0",
            "dynamic_features": {
                "derivative": 0.1,
                "rolling_std_1h": 0.05,  # Below variability threshold
            },
        }
        
        event = self.pipeline.process_event(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            ml_features=ml_features,
            regime="STABLE_NORMAL",
            anomaly_score=0.5,  # Below threshold
        )
        
        self.assertIsNone(event)
    
    def test_process_event_duplicate(self):
        """Test processing duplicate event (should be filtered)."""
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
        
        # First event
        self.pipeline.process_event(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            ml_features=ml_features,
            regime="STARTUP",
            anomaly_score=0.85,
        )
        
        # Duplicate event (same timestamp)
        event = self.pipeline.process_event(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            ml_features=ml_features,
            regime="STARTUP",
            anomaly_score=0.85,
        )
        
        self.assertIsNone(event)
    
    def test_process_event_memory_disabled(self):
        """Test processing event when memory is disabled."""
        self.registry.enable_memory_storage(False)
        
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
        
        event = self.pipeline.process_event(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            ml_features=ml_features,
            regime="STARTUP",
            anomaly_score=0.85,
        )
        
        self.assertIsNone(event)
    
    def test_clear_dedup_cache(self):
        """Test clearing deduplication cache."""
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
        
        # Process event
        self.pipeline.process_event(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            ml_features=ml_features,
            regime="STARTUP",
            anomaly_score=0.85,
        )
        
        # Clear cache
        self.pipeline.clear_dedup_cache()
        
        # Same event should not be filtered now
        event = self.pipeline.process_event(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            ml_features=ml_features,
            regime="STARTUP",
            anomaly_score=0.85,
        )
        
        self.assertIsNotNone(event)


if __name__ == "__main__":
    unittest.main()
