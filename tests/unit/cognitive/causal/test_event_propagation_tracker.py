"""
Unit tests for EventPropagationTracker.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import unittest

from infrastructure.ml.cognitive.causal.event_propagation_tracker import EventPropagationTracker


class TestEventPropagationTracker(unittest.TestCase):
    """Test cases for EventPropagationTracker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = EventPropagationTracker()
    
    def test_start_propagation(self):
        """Test starting a new propagation."""
        propagation_id = self.tracker.start_propagation(
            source_sensor_id=12345,
            timestamp=1234567890.0,
        )
        
        self.assertIsNotNone(propagation_id)
        self.assertIn(propagation_id, self.tracker._active_propagations)
    
    def test_add_to_propagation(self):
        """Test adding target sensor to propagation."""
        propagation_id = self.tracker.start_propagation(
            source_sensor_id=12345,
            timestamp=1234567890.0,
        )
        
        self.tracker.add_to_propagation(
            propagation_id=propagation_id,
            target_sensor_id=67890,
            timestamp=1234567891.0,
        )
        
        propagation = self.tracker._active_propagations[propagation_id]
        self.assertIn(67890, propagation["target_sensors"])
    
    def test_end_propagation(self):
        """Test ending propagation tracking."""
        propagation_id = self.tracker.start_propagation(
            source_sensor_id=12345,
            timestamp=1234567890.0,
        )
        
        self.tracker.add_to_propagation(
            propagation_id=propagation_id,
            target_sensor_id=67890,
            timestamp=1234567891.0,
        )
        
        event = self.tracker.end_propagation(
            propagation_id=propagation_id,
            end_timestamp=1234567892.0,
        )
        
        self.assertIsNotNone(event)
        self.assertEqual(event.source_sensor_id, 12345)
        self.assertEqual(event.propagation_duration_seconds, 2.0)
    
    def test_get_propagation_statistics(self):
        """Test getting propagation statistics."""
        propagation_id = self.tracker.start_propagation(
            source_sensor_id=12345,
            timestamp=1234567890.0,
        )
        
        self.tracker.add_to_propagation(
            propagation_id=propagation_id,
            target_sensor_id=67890,
            timestamp=1234567891.0,
        )
        
        self.tracker.end_propagation(
            propagation_id=propagation_id,
            end_timestamp=1234567892.0,
        )
        
        stats = self.tracker.get_propagation_statistics(12345, 67890)
        
        self.assertEqual(stats["count"], 1)
        self.assertEqual(stats["avg_duration_seconds"], 2.0)
    
    def test_get_active_propagations(self):
        """Test getting active propagations."""
        propagation_id = self.tracker.start_propagation(
            source_sensor_id=12345,
            timestamp=1234567890.0,
        )
        
        active = self.tracker.get_active_propagations()
        
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["propagation_id"], propagation_id)
    
    def test_get_completed_propagations(self):
        """Test getting completed propagations."""
        propagation_id = self.tracker.start_propagation(
            source_sensor_id=12345,
            timestamp=1234567890.0,
        )
        
        self.tracker.add_to_propagation(
            propagation_id=propagation_id,
            target_sensor_id=67890,
            timestamp=1234567891.0,
        )
        
        self.tracker.end_propagation(
            propagation_id=propagation_id,
            end_timestamp=1234567892.0,
        )
        
        completed = self.tracker.get_completed_propagations()
        
        self.assertEqual(len(completed), 1)
    
    def test_calculate_propagation_confidence(self):
        """Test propagation confidence calculation via PropagationConfidenceCalculator."""
        from infrastructure.ml.cognitive.causal.utils.propagation_confidence import (
            PropagationConfidenceCalculator,
        )
        confidence = PropagationConfidenceCalculator.calculate(
            target_count=2,
            duration=10.0,
            max_window_seconds=300.0,
        )
        
        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_is_cascade(self):
        """Test cascade detection."""
        propagation_id = self.tracker.start_propagation(
            source_sensor_id=12345,
            timestamp=1234567890.0,
        )
        
        # Add more than 2 targets
        for target_id in [67890, 11111, 22222]:
            self.tracker.add_to_propagation(
                propagation_id=propagation_id,
                target_sensor_id=target_id,
                timestamp=1234567891.0,
            )
        
        event = self.tracker.end_propagation(
            propagation_id=propagation_id,
            end_timestamp=1234567892.0,
        )
        
        self.assertTrue(event.is_cascade)
    
    def test_reset(self):
        """Test resetting tracker."""
        propagation_id = self.tracker.start_propagation(
            source_sensor_id=12345,
            timestamp=1234567890.0,
        )
        
        self.tracker.reset()
        
        self.assertEqual(len(self.tracker._active_propagations), 0)
        self.assertEqual(len(self.tracker._completed_propagations), 0)


if __name__ == "__main__":
    unittest.main()
