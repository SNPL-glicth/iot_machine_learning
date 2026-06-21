"""
Unit tests for SemanticEventBuilder.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import unittest
from infrastructure.ml.cognitive.memory.semantic_event_builder import SemanticEventBuilder
from domain.entities.memory import MemoryEvent


class TestSemanticEventBuilder(unittest.TestCase):
    """Test cases for SemanticEventBuilder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.builder = SemanticEventBuilder()
    
    def test_build_anomaly_event(self):
        """Test building an anomaly event."""
        ml_features = {
            "timestamp": 1234567890.0,
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
        
        event = self.builder.build(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            ml_features=ml_features,
            regime="STARTUP",
            anomaly_score=0.85,
        )
        
        self.assertIsInstance(event, MemoryEvent)
        self.assertEqual(event.sensor_id, 12345)
        self.assertEqual(event.sensor_type, "TEMPERATURE")
        self.assertEqual(event.regime, "STARTUP")
        self.assertEqual(event.anomaly_score, 0.85)
        self.assertEqual(event.event_type, "ANOMALY_CONFIRMED")
        self.assertIn("Sensor 12345", event.semantic_text)
        self.assertIn("STARTUP", event.semantic_text)
    
    def test_build_regime_transition_event(self):
        """Test building a regime transition event."""
        ml_features = {
            "timestamp": 1234567890.0,
            "current_value": 50.0,
            "baseline": 48.0,
            "z_score": 0.5,
            "trend": "stable",
            "stability": 0.9,
            "model_version": "2.0.0",
            "dynamic_features": {},
        }
        
        event = self.builder.build(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            ml_features=ml_features,
            regime="STABLE_NORMAL",
            anomaly_score=0.3,
            previous_regime="STARTUP",
            transition_duration=900.0,
        )
        
        self.assertEqual(event.event_type, "REGIME_TRANSITION")
        self.assertEqual(event.metadata["previous_regime"], "STARTUP")
        self.assertEqual(event.metadata["transition_duration"], 900.0)
    
    def test_build_operational_state_event(self):
        """Test building an operational state event."""
        ml_features = {
            "timestamp": 1234567890.0,
            "current_value": 45.0,
            "baseline": 44.0,
            "z_score": 0.3,
            "trend": "stable",
            "stability": 0.95,
            "model_version": "2.0.0",
            "dynamic_features": {},
        }
        
        event = self.builder.build(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            ml_features=ml_features,
            regime="STABLE_NORMAL",
            anomaly_score=0.2,
        )
        
        self.assertEqual(event.event_type, "OPERATIONAL_STATE")
    
    def test_semantic_text_generation(self):
        """Test semantic text generation."""
        ml_features = {
            "timestamp": 1234567890.0,
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
        
        event = self.builder.build(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            ml_features=ml_features,
            regime="STARTUP",
            anomaly_score=0.85,
        )
        
        self.assertIn("Sensor 12345", event.semantic_text)
        self.assertIn("TEMPERATURE", event.semantic_text)
        self.assertIn("STARTUP", event.semantic_text)
        self.assertIn("85.20", event.semantic_text)
        self.assertIn("45.00", event.semantic_text)
        self.assertIn("3.20", event.semantic_text)
        self.assertIn("0.85", event.semantic_text)
        self.assertIn("2.50", event.semantic_text)
        self.assertIn("8.50", event.semantic_text)
    
    def test_metadata_structure(self):
        """Test metadata structure."""
        ml_features = {
            "timestamp": 1234567890.0,
            "current_value": 85.2,
            "baseline": 45.0,
            "z_score": 3.2,
            "trend": "increasing",
            "stability": 0.5,
            "model_version": "2.0.0",
            "dynamic_features": {},
        }
        
        event = self.builder.build(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            ml_features=ml_features,
            regime="STARTUP",
            anomaly_score=0.85,
        )
        
        self.assertIn("value", event.metadata)
        self.assertIn("baseline", event.metadata)
        self.assertIn("z_score", event.metadata)
        self.assertIn("trend", event.metadata)
        self.assertIn("stability", event.metadata)
        self.assertIn("model_version", event.metadata)
        
        self.assertEqual(event.metadata["value"], 85.2)
        self.assertEqual(event.metadata["baseline"], 45.0)
        self.assertEqual(event.metadata["z_score"], 3.2)
        self.assertEqual(event.metadata["trend"], "increasing")
        self.assertEqual(event.metadata["stability"], 0.5)
        self.assertEqual(event.metadata["model_version"], "2.0.0")


if __name__ == "__main__":
    unittest.main()
