"""
Unit tests for CognitiveMetricsCollector.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import unittest

from infrastructure.ml.cognitive.observability.cognitive_metrics_collector import CognitiveMetricsCollector


class TestCognitiveMetricsCollector(unittest.TestCase):
    """Test cases for CognitiveMetricsCollector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collector = CognitiveMetricsCollector()
    
    def test_record_regime(self):
        """Test recording regime."""
        self.collector.record_regime("STARTUP")
        self.collector.record_regime("STARTUP")
        self.collector.record_regime("STABLE_NORMAL")
        
        metrics = self.collector.collect_metrics()
        
        self.assertEqual(metrics.regime_distribution["STARTUP"], 2)
        self.assertEqual(metrics.regime_distribution["STABLE_NORMAL"], 1)
    
    def test_record_anomaly(self):
        """Test recording anomaly."""
        self.collector.record_anomaly("ANOMALY_CONFIRMED")
        self.collector.record_anomaly("ANOMALY_SUSPECTED")
        
        metrics = self.collector.collect_metrics()
        
        self.assertEqual(metrics.anomaly_distribution["ANOMALY_CONFIRMED"], 1)
        self.assertEqual(metrics.anomaly_distribution["ANOMALY_SUSPECTED"], 1)
    
    def test_record_retrieval_hit(self):
        """Test recording retrieval hit."""
        self.collector.record_retrieval(hit=True, similarity=0.85)
        self.collector.record_retrieval(hit=True, similarity=0.90)
        self.collector.record_retrieval(hit=False)
        
        metrics = self.collector.collect_metrics()
        
        self.assertEqual(metrics.retrieval_hit_rate, 2/3)
        self.assertAlmostEqual(metrics.retrieval_similarity_mean, 0.875, places=2)
    
    def test_record_explainability_consistency(self):
        """Test recording explainability consistency."""
        self.collector.record_explainability_consistency(0.9)
        self.collector.record_explainability_consistency(0.8)
        
        metrics = self.collector.collect_metrics()
        
        self.assertAlmostEqual(metrics.explainability_consistency, 0.85, places=2)
    
    def test_record_confidence(self):
        """Test recording confidence."""
        self.collector.record_confidence(0.9)
        self.collector.record_confidence(0.5)
        self.collector.record_confidence(0.3)
        
        metrics = self.collector.collect_metrics()
        
        self.assertEqual(metrics.confidence_distribution["high"], 1/3)
        self.assertEqual(metrics.confidence_distribution["medium"], 1/3)
        self.assertEqual(metrics.confidence_distribution["low"], 1/3)
    
    def test_record_memory_growth(self):
        """Test recording memory growth."""
        self.collector.record_memory_growth(0.1)
        self.collector.record_memory_growth(0.2)
        
        metrics = self.collector.collect_metrics()
        
        self.assertAlmostEqual(metrics.memory_growth_rate, 0.15, places=2)
    
    def test_reset(self):
        """Test resetting metrics."""
        self.collector.record_regime("STARTUP")
        self.collector.record_anomaly("ANOMALY_CONFIRMED")
        
        self.collector.reset()
        
        metrics = self.collector.collect_metrics()
        
        self.assertEqual(metrics.regime_distribution, {})
        self.assertEqual(metrics.anomaly_distribution, {})


if __name__ == "__main__":
    unittest.main()
