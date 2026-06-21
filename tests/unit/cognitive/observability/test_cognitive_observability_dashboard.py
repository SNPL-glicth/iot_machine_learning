"""
Unit tests for CognitiveObservabilityDashboard.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import unittest
from unittest.mock import Mock

from infrastructure.ml.cognitive.observability.cognitive_observability_dashboard import CognitiveObservabilityDashboard
from infrastructure.ml.cognitive.observability.cognitive_metrics_collector import CognitiveMetricsCollector
from infrastructure.ml.cognitive.observability.memory_health_monitor import MemoryHealthMonitor
from infrastructure.ml.cognitive.observability.drift_detection_engine import DriftDetectionEngine
from infrastructure.ml.cognitive.observability.explainability_validator import ExplainabilityValidator
from infrastructure.ml.cognitive.observability.feedback_loop_manager import FeedbackLoopManager


class TestCognitiveObservabilityDashboard(unittest.TestCase):
    """Test cases for CognitiveObservabilityDashboard."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics_collector = CognitiveMetricsCollector()
        self.memory_health_monitor = MemoryHealthMonitor()
        self.drift_detection_engine = DriftDetectionEngine()
        self.explainability_validator = ExplainabilityValidator()
        self.feedback_loop_manager = FeedbackLoopManager()
        
        self.dashboard = CognitiveObservabilityDashboard(
            metrics_collector=self.metrics_collector,
            memory_health_monitor=self.memory_health_monitor,
            drift_detection_engine=self.drift_detection_engine,
            explainability_validator=self.explainability_validator,
            feedback_loop_manager=self.feedback_loop_manager,
        )
    
    def test_get_observability_summary(self):
        """Test getting observability summary."""
        # Record some metrics
        self.metrics_collector.record_regime("STARTUP")
        self.metrics_collector.record_anomaly("ANOMALY_CONFIRMED")
        self.metrics_collector.record_retrieval(hit=True, similarity=0.85)
        
        summary = self.dashboard.get_observability_summary()
        
        self.assertIn("timestamp", summary)
        self.assertIn("cognitive_metrics", summary)
        self.assertIn("memory_health", summary)
        self.assertIn("feedback_summary", summary)
        self.assertIn("health_status", summary)
    
    def test_get_regime_distribution(self):
        """Test getting regime distribution."""
        self.metrics_collector.record_regime("STARTUP")
        self.metrics_collector.record_regime("STARTUP")
        self.metrics_collector.record_regime("STABLE_NORMAL")
        
        distribution = self.dashboard.get_regime_distribution()
        
        self.assertEqual(distribution["STARTUP"], 2)
        self.assertEqual(distribution["STABLE_NORMAL"], 1)
    
    def test_get_anomaly_distribution(self):
        """Test getting anomaly distribution."""
        self.metrics_collector.record_anomaly("ANOMALY_CONFIRMED")
        self.metrics_collector.record_anomaly("ANOMALY_SUSPECTED")
        
        distribution = self.dashboard.get_anomaly_distribution()
        
        self.assertEqual(distribution["ANOMALY_CONFIRMED"], 1)
        self.assertEqual(distribution["ANOMALY_SUSPECTED"], 1)
    
    def test_get_retrieval_metrics(self):
        """Test getting retrieval metrics."""
        self.metrics_collector.record_retrieval(hit=True, similarity=0.85)
        self.metrics_collector.record_retrieval(hit=False)
        
        metrics = self.dashboard.get_retrieval_metrics()
        
        self.assertEqual(metrics["hit_rate"], 0.5)
        self.assertEqual(metrics["similarity_mean"], 0.85)
    
    def test_get_explainability_metrics(self):
        """Test getting explainability metrics."""
        self.metrics_collector.record_explainability_consistency(0.9)
        
        metrics = self.dashboard.get_explainability_metrics()
        
        self.assertEqual(metrics["consistency"], 0.9)
    
    def test_get_confidence_distribution(self):
        """Test getting confidence distribution."""
        self.metrics_collector.record_confidence(0.9)
        self.metrics_collector.record_confidence(0.5)
        self.metrics_collector.record_confidence(0.3)
        
        distribution = self.dashboard.get_confidence_distribution()
        
        self.assertEqual(distribution["high"], 1/3)
        self.assertEqual(distribution["medium"], 1/3)
        self.assertEqual(distribution["low"], 1/3)
    
    def test_get_memory_metrics(self):
        """Test getting memory metrics."""
        self.metrics_collector.record_memory_growth(0.1)
        self.metrics_collector.record_ttl_cleanup(0.05)
        
        metrics = self.dashboard.get_memory_metrics()
        
        self.assertEqual(metrics["growth_rate"], 0.1)
        self.assertEqual(metrics["ttl_cleanup_rate"], 0.05)
    
    def test_get_contextual_confidence_calibration(self):
        """Test getting contextual confidence calibration."""
        self.metrics_collector.record_contextual_confidence(0.85)
        
        calibration = self.dashboard.get_contextual_confidence_calibration()
        
        self.assertEqual(calibration, 0.85)
    
    def test_calculate_health_status_healthy(self):
        """Test calculating healthy status."""
        # Record healthy metrics
        self.metrics_collector.record_retrieval(hit=True, similarity=0.85)
        self.metrics_collector.record_explainability_consistency(0.9)
        self.memory_health_monitor.record_memory_explosion_risk(0.1)
        
        status = self.dashboard._calculate_health_status(
            self.metrics_collector.collect_metrics(),
            self.memory_health_monitor.assess_health(),
        )
        
        self.assertEqual(status, "HEALTHY")
    
    def test_calculate_health_status_warning(self):
        """Test calculating warning status."""
        # Record warning metrics
        self.metrics_collector.record_retrieval(hit=False)
        self.metrics_collector.record_explainability_consistency(0.6)
        self.memory_health_monitor.record_memory_explosion_risk(0.3)
        
        status = self.dashboard._calculate_health_status(
            self.metrics_collector.collect_metrics(),
            self.memory_health_monitor.assess_health(),
        )
        
        self.assertEqual(status, "WARNING")
    
    def test_calculate_health_status_critical(self):
        """Test calculating critical status."""
        # Record critical metrics
        self.memory_health_monitor.record_memory_explosion_risk(0.8)
        
        status = self.dashboard._calculate_health_status(
            self.metrics_collector.collect_metrics(),
            self.memory_health_monitor.assess_health(),
        )
        
        self.assertEqual(status, "CRITICAL")


if __name__ == "__main__":
    unittest.main()
