"""
Unit tests for CognitiveMemoryRegistry.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import unittest
from infrastructure.ml.cognitive.memory.cognitive_memory_registry import CognitiveMemoryRegistry


class TestCognitiveMemoryRegistry(unittest.TestCase):
    """Test cases for CognitiveMemoryRegistry."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = CognitiveMemoryRegistry()
    
    def test_default_ttl_configuration(self):
        """Test default TTL configuration."""
        self.assertEqual(
            self.registry.get_ttl("ANOMALY_CONFIRMED"),
            365 * 24 * 3600  # 1 año
        )
        self.assertEqual(
            self.registry.get_ttl("ANOMALY_SUSPECTED"),
            7 * 24 * 3600  # 1 semana
        )
        self.assertEqual(
            self.registry.get_ttl("REGIME_TRANSITION"),
            180 * 24 * 3600  # 6 meses
        )
    
    def test_get_ttl_unknown_event_type(self):
        """Test TTL for unknown event type returns default."""
        ttl = self.registry.get_ttl("UNKNOWN_EVENT")
        self.assertEqual(ttl, 7 * 24 * 3600)  # Default: 1 semana
    
    def test_set_ttl(self):
        """Test setting TTL for event type."""
        self.registry.set_ttl("ANOMALY_CONFIRMED", 30 * 24 * 3600)  # 30 días
        self.assertEqual(
            self.registry.get_ttl("ANOMALY_CONFIRMED"),
            30 * 24 * 3600
        )
    
    def test_min_anomaly_score(self):
        """Test minimum anomaly score threshold."""
        self.assertEqual(self.registry.min_anomaly_score, 0.6)
    
    def test_min_feature_variability(self):
        """Test minimum feature variability threshold."""
        self.assertEqual(self.registry.min_feature_variability, 0.1)
    
    def test_enable_memory_storage(self):
        """Test enabling/disabling memory storage."""
        self.assertTrue(self.registry.enable_memory)
        
        self.registry.enable_memory_storage(False)
        self.assertFalse(self.registry.enable_memory)
        
        self.registry.enable_memory_storage(True)
        self.assertTrue(self.registry.enable_memory)
    
    def test_enable_retrieval_feature(self):
        """Test enabling/disabling retrieval."""
        self.assertTrue(self.registry.enable_retrieval)
        
        self.registry.enable_retrieval_feature(False)
        self.assertFalse(self.registry.enable_retrieval)
        
        self.registry.enable_retrieval_feature(True)
        self.assertTrue(self.registry.enable_retrieval)
    
    def test_get_ttl_config(self):
        """Test getting all TTL configuration."""
        ttl_config = self.registry.get_ttl_config()
        
        self.assertIsInstance(ttl_config, dict)
        self.assertIn("ANOMALY_CONFIRMED", ttl_config)
        self.assertIn("ANOMALY_SUSPECTED", ttl_config)
        self.assertIn("REGIME_TRANSITION", ttl_config)
        
        # Verify it's a copy, not reference
        ttl_config["ANOMALY_CONFIRMED"] = 999
        self.assertNotEqual(
            self.registry.get_ttl("ANOMALY_CONFIRMED"),
            999
        )


if __name__ == "__main__":
    unittest.main()
