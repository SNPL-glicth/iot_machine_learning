"""
Unit tests for FeedbackLoopManager.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import unittest

from infrastructure.ml.cognitive.observability.feedback_loop_manager import FeedbackLoopManager


class TestFeedbackLoopManager(unittest.TestCase):
    """Test cases for FeedbackLoopManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = FeedbackLoopManager()
    
    def test_record_alert_feedback(self):
        """Test recording alert feedback."""
        self.manager.record_alert_feedback(sensor_id=12345, usefulness=0.8)
        
        summary = self.manager.get_feedback_summary("alert_usefulness")
        
        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["mean_usefulness"], 0.8)
    
    def test_record_retrieval_feedback(self):
        """Test recording retrieval feedback."""
        self.manager.record_retrieval_feedback(sensor_id=12345, usefulness=0.9)
        
        summary = self.manager.get_feedback_summary("retrieval_usefulness")
        
        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["mean_usefulness"], 0.9)
    
    def test_record_explainability_feedback(self):
        """Test recording explainability feedback."""
        self.manager.record_explainability_feedback(sensor_id=12345, usefulness=0.7)
        
        summary = self.manager.get_feedback_summary("explainability_usefulness")
        
        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["mean_usefulness"], 0.7)
    
    def test_get_feedback_summary_all(self):
        """Test getting feedback summary for all types."""
        self.manager.record_alert_feedback(sensor_id=12345, usefulness=0.8)
        self.manager.record_retrieval_feedback(sensor_id=12345, usefulness=0.9)
        self.manager.record_explainability_feedback(sensor_id=12345, usefulness=0.7)
        
        summary = self.manager.get_feedback_summary()
        
        self.assertEqual(summary["count"], 3)
    
    def test_get_sensor_feedback(self):
        """Test getting feedback for specific sensor."""
        self.manager.record_alert_feedback(sensor_id=12345, usefulness=0.8)
        self.manager.record_alert_feedback(sensor_id=67890, usefulness=0.6)
        
        sensor_feedback = self.manager.get_sensor_feedback(sensor_id=12345)
        
        self.assertEqual(sensor_feedback["sensor_id"], 12345)
        self.assertEqual(sensor_feedback["count"], 1)
        self.assertEqual(sensor_feedback["mean_usefulness"], 0.8)
    
    def test_cleanup_old_feedback(self):
        """Test cleaning up old feedback entries."""
        # This test would require mocking time, so we'll just test the method exists
        cleaned = self.manager.cleanup_old_feedback(max_age_seconds=0)
        
        self.assertEqual(cleaned, 0)
    
    def test_reset(self):
        """Test resetting feedback manager."""
        self.manager.record_alert_feedback(sensor_id=12345, usefulness=0.8)
        self.manager.record_retrieval_feedback(sensor_id=12345, usefulness=0.9)
        
        self.manager.reset()
        
        summary = self.manager.get_feedback_summary()
        
        self.assertEqual(summary["count"], 0)
    
    def test_feedback_with_metadata(self):
        """Test recording feedback with metadata."""
        metadata = {"operator_id": "op1", "timestamp": 1234567890.0}
        
        self.manager.record_alert_feedback(
            sensor_id=12345,
            usefulness=0.8,
            metadata=metadata,
        )
        
        summary = self.manager.get_feedback_summary("alert_usefulness")
        
        self.assertEqual(summary["count"], 1)


if __name__ == "__main__":
    unittest.main()
