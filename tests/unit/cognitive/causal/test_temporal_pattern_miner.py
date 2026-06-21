"""
Unit tests for TemporalPatternMiner.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import unittest

from infrastructure.ml.cognitive.causal.temporal_pattern_miner import TemporalPatternMiner


class TestTemporalPatternMiner(unittest.TestCase):
    """Test cases for TemporalPatternMiner."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.miner = TemporalPatternMiner(min_support=2)
    
    def test_add_event_sequence(self):
        """Test adding event sequence."""
        sensor_sequence = [12345, 67890, 11111]
        timestamps = [1234567890.0, 1234567891.0, 1234567892.0]
        
        self.miner.add_event_sequence(sensor_sequence, timestamps)
        
        self.assertEqual(len(self.miner._event_sequences), 1)
    
    def test_mine_patterns_no_data(self):
        """Test pattern mining with no data."""
        patterns = self.miner.mine_patterns()
        
        self.assertEqual(len(patterns), 0)
    
    def test_mine_patterns_with_data(self):
        """Test pattern mining with data."""
        # Add repeated sequence
        for i in range(5):
            sensor_sequence = [12345, 67890, 11111]
            timestamps = [1234567890.0 + i * 10, 1234567891.0 + i * 10, 1234567892.0 + i * 10]
            self.miner.add_event_sequence(sensor_sequence, timestamps)
        
        patterns = self.miner.mine_patterns()
        
        self.assertGreater(len(patterns), 0)
    
    def test_mine_frequent_sequences(self):
        """Test mining frequent sequences."""
        # Add repeated sequence
        for i in range(5):
            sensor_sequence = [12345, 67890, 11111]
            timestamps = [1234567890.0 + i * 10, 1234567891.0 + i * 10, 1234567892.0 + i * 10]
            self.miner.add_event_sequence(sensor_sequence, timestamps)
        
        frequent_sequences = self.miner._mine_frequent_sequences()
        
        self.assertGreater(len(frequent_sequences), 0)
    
    def test_detect_transitional_chains(self):
        """Test detecting transitional chains."""
        # Add sequences with common transitions
        for i in range(5):
            sensor_sequence = [12345, 67890, 11111]
            timestamps = [1234567890.0 + i * 10, 1234567891.0 + i * 10, 1234567892.0 + i * 10]
            self.miner.add_event_sequence(sensor_sequence, timestamps)
        
        chains = self.miner.detect_transitional_chains()
        
        self.assertGreater(len(chains), 0)
    
    def test_find_operational_motifs(self):
        """Test finding operational motifs."""
        # Add sequences of length 3-5
        for i in range(5):
            sensor_sequence = [12345, 67890, 11111, 22222]
            timestamps = [1234567890.0 + i * 10, 1234567891.0 + i * 10, 1234567892.0 + i * 10, 1234567893.0 + i * 10]
            self.miner.add_event_sequence(sensor_sequence, timestamps)
        
        motifs = self.miner.find_operational_motifs()
        
        self.assertGreater(len(motifs), 0)
    
    def test_calculate_pattern_confidence(self):
        """Test pattern confidence calculation."""
        sequence = [12345, 67890, 11111]
        frequency = 5
        
        confidence = self.miner._calculate_pattern_confidence(sequence, frequency)
        
        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_reset(self):
        """Test resetting miner."""
        sensor_sequence = [12345, 67890, 11111]
        timestamps = [1234567890.0, 1234567891.0, 1234567892.0]
        
        self.miner.add_event_sequence(sensor_sequence, timestamps)
        self.miner.reset()
        
        self.assertEqual(len(self.miner._event_sequences), 0)


if __name__ == "__main__":
    unittest.main()
