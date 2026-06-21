"""
Unit tests for OperationalSequenceRegistry.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import unittest

from infrastructure.ml.cognitive.causal.operational_sequence_registry import OperationalSequenceRegistry
from domain.entities.causal import TemporalPattern


class TestOperationalSequenceRegistry(unittest.TestCase):
    """Test cases for OperationalSequenceRegistry."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = OperationalSequenceRegistry()
    
    def test_register_sequence(self):
        """Test registering a sequence."""
        pattern = TemporalPattern(
            pattern_id="pattern_1",
            sequence=[12345, 67890, 11111],
            frequency=5,
            avg_duration_seconds=10.0,
            confidence=0.85,
            is_pre_anomaly=False,
            timestamp=1234567890.0,
        )
        
        self.registry.register_sequence(pattern)
        
        self.assertIn("pattern_1", self.registry._sequences)
    
    def test_get_sequence(self):
        """Test getting a sequence by ID."""
        pattern = TemporalPattern(
            pattern_id="pattern_1",
            sequence=[12345, 67890, 11111],
            frequency=5,
            avg_duration_seconds=10.0,
            confidence=0.85,
            is_pre_anomaly=False,
            timestamp=1234567890.0,
        )
        
        self.registry.register_sequence(pattern)
        
        retrieved = self.registry.get_sequence("pattern_1")
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.pattern_id, "pattern_1")
    
    def test_get_frequent_sequences(self):
        """Test getting frequent sequences."""
        pattern = TemporalPattern(
            pattern_id="pattern_1",
            sequence=[12345, 67890, 11111],
            frequency=10,
            avg_duration_seconds=10.0,
            confidence=0.85,
            is_pre_anomaly=False,
            timestamp=1234567890.0,
        )
        
        for _ in range(5):
            self.registry.register_sequence(pattern)
        
        frequent = self.registry.get_frequent_sequences(min_frequency=5)
        
        self.assertEqual(len(frequent), 1)
        self.assertEqual(frequent[0].pattern_id, "pattern_1")
    
    def test_get_anomaly_precursors(self):
        """Test getting anomaly precursors."""
        pattern = TemporalPattern(
            pattern_id="pattern_1",
            sequence=[12345, 67890, 11111],
            frequency=5,
            avg_duration_seconds=10.0,
            confidence=0.85,
            is_pre_anomaly=True,
            timestamp=1234567890.0,
        )
        
        self.registry.register_sequence(pattern)
        
        precursors = self.registry.get_anomaly_precursors()
        
        self.assertEqual(len(precursors), 1)
        self.assertTrue(precursors[0].is_pre_anomaly)
    
    def test_find_matching_sequences(self):
        """Test finding matching sequences."""
        pattern = TemporalPattern(
            pattern_id="pattern_1",
            sequence=[12345, 67890],
            frequency=5,
            avg_duration_seconds=10.0,
            confidence=0.85,
            is_pre_anomaly=False,
            timestamp=1234567890.0,
        )
        
        self.registry.register_sequence(pattern)
        
        matching = self.registry.find_matching_sequences([12345, 67890, 11111])
        
        self.assertEqual(len(matching), 1)
    
    def test_get_operational_chains(self):
        """Test getting operational chains."""
        pattern = TemporalPattern(
            pattern_id="pattern_1",
            sequence=[12345, 67890, 11111],
            frequency=5,
            avg_duration_seconds=10.0,
            confidence=0.85,
            is_pre_anomaly=False,
            timestamp=1234567890.0,
        )
        
        self.registry.register_sequence(pattern)
        
        chains = self.registry.get_operational_chains(min_length=3)
        
        self.assertEqual(len(chains), 1)
    
    def test_get_sequence_statistics(self):
        """Test getting sequence statistics."""
        pattern = TemporalPattern(
            pattern_id="pattern_1",
            sequence=[12345, 67890, 11111],
            frequency=5,
            avg_duration_seconds=10.0,
            confidence=0.85,
            is_pre_anomaly=False,
            timestamp=1234567890.0,
        )
        
        self.registry.register_sequence(pattern)
        
        stats = self.registry.get_sequence_statistics()
        
        self.assertEqual(stats["total_sequences"], 1)
        self.assertEqual(stats["average_frequency"], 5.0)
        self.assertEqual(stats["average_confidence"], 0.85)
    
    def test_cleanup_old_sequences(self):
        """Test cleaning up old sequences."""
        pattern = TemporalPattern(
            pattern_id="pattern_1",
            sequence=[12345, 67890, 11111],
            frequency=5,
            avg_duration_seconds=10.0,
            confidence=0.85,
            is_pre_anomaly=False,
            timestamp=1234567890.0 - 86400 * 8,  # 8 days ago
        )
        
        self.registry.register_sequence(pattern)
        
        cleaned = self.registry.cleanup_old_sequences(max_age_seconds=86400 * 7)
        
        self.assertEqual(cleaned, 1)
    
    def test_reset(self):
        """Test resetting registry."""
        pattern = TemporalPattern(
            pattern_id="pattern_1",
            sequence=[12345, 67890, 11111],
            frequency=5,
            avg_duration_seconds=10.0,
            confidence=0.85,
            is_pre_anomaly=False,
            timestamp=1234567890.0,
        )
        
        self.registry.register_sequence(pattern)
        self.registry.reset()
        
        self.assertEqual(len(self.registry._sequences), 0)


if __name__ == "__main__":
    unittest.main()
