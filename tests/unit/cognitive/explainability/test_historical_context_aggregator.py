"""
Unit tests for HistoricalContextAggregator.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import unittest

from infrastructure.ml.cognitive.explainability.historical_context_aggregator import HistoricalContextAggregator
from domain.entities.memory import MemoryEvent


class TestHistoricalContextAggregator(unittest.TestCase):
    """Test cases for HistoricalContextAggregator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.aggregator = HistoricalContextAggregator()
    
    def test_aggregate_empty_events(self):
        """Test aggregating empty events."""
        result = self.aggregator.aggregate([])
        
        self.assertEqual(result["similar_event_count"], 0)
        self.assertIn("No historical similar events", result["historical_context"])
        self.assertEqual(result["historical_patterns"], [])
    
    def test_aggregate_single_event(self):
        """Test aggregating single event."""
        event = MemoryEvent(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            timestamp=1234567890.0,
            event_type="ANOMALY_CONFIRMED",
            semantic_text="Sensor 12345...",
            regime="STARTUP",
            anomaly_score=0.85,
            dynamic_features={"derivative": 2.5},
            metadata={"value": 85.2},
        )
        
        result = self.aggregator.aggregate([event])
        
        self.assertEqual(result["similar_event_count"], 1)
        self.assertIn("1 evento similar", result["historical_context"])
        self.assertIn("STARTUP", result["historical_context"])
    
    def test_aggregate_multiple_events(self):
        """Test aggregating multiple events."""
        events = [
            MemoryEvent(
                sensor_id=12345,
                sensor_type="TEMPERATURE",
                timestamp=1234567890.0 + i,
                event_type="ANOMALY_CONFIRMED",
                semantic_text="Sensor 12345...",
                regime="STARTUP" if i % 2 == 0 else "STABLE_NORMAL",
                anomaly_score=0.8 + i * 0.02,
                dynamic_features={},
                metadata={"value": 85.0 + i},
            )
            for i in range(5)
        ]
        
        result = self.aggregator.aggregate(events)
        
        self.assertEqual(result["similar_event_count"], 5)
        self.assertIn("5 eventos similares", result["historical_context"])
        self.assertIn("STARTUP", result["historical_context"])
    
    def test_regime_distribution(self):
        """Test regime distribution calculation."""
        events = [
            MemoryEvent(
                sensor_id=12345,
                sensor_type="TEMPERATURE",
                timestamp=1234567890.0 + i,
                event_type="ANOMALY_CONFIRMED",
                semantic_text="Sensor 12345...",
                regime="STARTUP" if i < 3 else "STABLE_NORMAL",
                anomaly_score=0.8,
                dynamic_features={},
                metadata={},
            )
            for i in range(5)
        ]
        
        result = self.aggregator.aggregate(events)
        
        self.assertIn("regime_distribution", result)
        self.assertEqual(result["regime_distribution"]["STARTUP"], 3)
        self.assertEqual(result["regime_distribution"]["STABLE_NORMAL"], 2)
    
    def test_anomaly_score_distribution(self):
        """Test anomaly score distribution calculation."""
        events = [
            MemoryEvent(
                sensor_id=12345,
                sensor_type="TEMPERATURE",
                timestamp=1234567890.0 + i,
                event_type="ANOMALY_CONFIRMED",
                semantic_text="Sensor 12345...",
                regime="STARTUP",
                anomaly_score=0.7 + i * 0.05,
                dynamic_features={},
                metadata={},
            )
            for i in range(5)
        ]
        
        result = self.aggregator.aggregate(events)
        
        self.assertIn("anomaly_score_distribution", result)
        self.assertIn("mean", result["anomaly_score_distribution"])
        self.assertIn("min", result["anomaly_score_distribution"])
        self.assertIn("max", result["anomaly_score_distribution"])
        
        # Check statistics
        self.assertAlmostEqual(result["anomaly_score_distribution"]["mean"], 0.8, places=2)
        self.assertEqual(result["anomaly_score_distribution"]["min"], 0.7)
        self.assertEqual(result["anomaly_score_distribution"]["max"], 0.9)
    
    def test_identify_patterns(self):
        """Test pattern identification."""
        events = [
            MemoryEvent(
                sensor_id=12345,
                sensor_type="TEMPERATURE",
                timestamp=1234567890.0 + i,
                event_type="ANOMALY_CONFIRMED",
                semantic_text="Sensor 12345...",
                regime="STARTUP",
                anomaly_score=0.8,
                dynamic_features={},
                metadata={},
            )
            for i in range(5)
        ]
        
        result = self.aggregator.aggregate(events)
        
        self.assertIn("historical_patterns", result)
        self.assertGreater(len(result["historical_patterns"]), 0)


if __name__ == "__main__":
    unittest.main()
