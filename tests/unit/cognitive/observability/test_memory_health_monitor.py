"""
Unit tests for MemoryHealthMonitor.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import unittest

from infrastructure.ml.cognitive.observability.memory_health_monitor import MemoryHealthMonitor


class TestMemoryHealthMonitor(unittest.TestCase):
    """Test cases for MemoryHealthMonitor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = MemoryHealthMonitor()
    
    def test_record_semantic_duplication(self):
        """Test recording semantic duplication."""
        self.monitor.record_semantic_duplication(0.1)
        self.monitor.record_semantic_duplication(0.15)
        
        health = self.monitor.assess_health()
        
        self.assertAlmostEqual(health.semantic_duplication_rate, 0.125, places=2)
    
    def test_assess_health_basic(self):
        """Test basic health assessment."""
        self.monitor.record_semantic_duplication(0.05)
        self.monitor.record_stale_memory(0.1)
        self.monitor.record_low_quality_memory(0.08)
        self.monitor.record_embedding_repetition(0.02)
        self.monitor.record_retrieval_degradation(0.1)
        self.monitor.record_memory_explosion_risk(0.2)
        
        health = self.monitor.assess_health()
        
        self.assertGreater(health.memory_quality_score, 0.5)
        self.assertGreater(health.retrieval_usefulness_score, 0.8)
        self.assertLess(health.memory_explosion_risk, 0.5)
    
    def test_assess_health_high_duplication(self):
        """Test health assessment with high duplication."""
        self.monitor.record_semantic_duplication(0.2)
        self.monitor.record_stale_memory(0.1)
        self.monitor.record_low_quality_memory(0.1)
        
        health = self.monitor.assess_health()
        
        self.assertLess(health.memory_quality_score, 0.8)
        self.assertIn("deduplicating", health.cleanup_recommendations[0].lower())
    
    def test_assess_memory_explosion_risk(self):
        """Test memory explosion risk detection."""
        self.monitor.record_memory_explosion_risk(0.8)
        
        health = self.monitor.assess_health()
        
        self.assertGreater(health.memory_explosion_risk, 0.7)
        self.assertTrue(any("explosion" in rec.lower() for rec in health.cleanup_recommendations))
    
    def test_calculate_memory_quality_score(self):
        """Test memory quality score calculation."""
        # High quality (low duplication, stale, low quality)
        self.monitor.record_semantic_duplication(0.01)
        self.monitor.record_stale_memory(0.02)
        self.monitor.record_low_quality_memory(0.01)
        
        health = self.monitor.assess_health()
        
        self.assertGreater(health.memory_quality_score, 0.95)
    
    def test_reset(self):
        """Test resetting monitor."""
        self.monitor.record_semantic_duplication(0.1)
        self.monitor.record_stale_memory(0.1)
        
        self.monitor.reset()
        
        health = self.monitor.assess_health()
        
        self.assertEqual(health.semantic_duplication_rate, 0.0)
        self.assertEqual(health.stale_memory_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
