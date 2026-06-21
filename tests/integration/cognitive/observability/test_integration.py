"""
Integration tests for cognitive observability with memory and explainability.

These tests integrate observability components with memory and explainability systems.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import unittest
from unittest.mock import Mock

from infrastructure.ml.cognitive.observability.cognitive_metrics_collector import CognitiveMetricsCollector
from infrastructure.ml.cognitive.observability.memory_health_monitor import MemoryHealthMonitor
from infrastructure.ml.cognitive.observability.drift_detection_engine import DriftDetectionEngine
from infrastructure.ml.cognitive.observability.explainability_validator import ExplainabilityValidator
from infrastructure.ml.cognitive.observability.feedback_loop_manager import FeedbackLoopManager
from infrastructure.ml.cognitive.observability.cognitive_observability_dashboard import CognitiveObservabilityDashboard
from infrastructure.ml.cognitive.memory.historical_similarity_retriever import HistoricalSimilarityRetriever
from infrastructure.ml.cognitive.memory.cognitive_memory_registry import CognitiveMemoryRegistry
from infrastructure.ml.cognitive.explainability.contextual_explainability_engine import ContextualExplainabilityEngine
from domain.entities.memory import MemoryEvent
from domain.entities.explainability import ContextualExplanation


class TestObservabilityIntegration(unittest.TestCase):
    """Integration tests for cognitive observability."""
    
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
        
        # Set up drift detection baselines
        self.drift_detection_engine.set_baselines(
            regime_distribution={"STARTUP": 0.3, "STABLE_NORMAL": 0.7},
            feature_means={"temperature": 50.0, "pressure": 100.0},
            anomaly_frequency=0.1,
            embedding_mean=0.5,
        )
    
    def test_observability_with_memory_operations(self):
        """Test observability during memory operations."""
        # Simulate memory operations
        self.metrics_collector.record_regime("STARTUP")
        self.metrics_collector.record_anomaly("ANOMALY_CONFIRMED")
        self.metrics_collector.record_retrieval(hit=True, similarity=0.85)
        self.metrics_collector.record_memory_growth(0.1)
        
        # Record memory health metrics
        self.memory_health_monitor.record_semantic_duplication(0.05)
        self.memory_health_monitor.record_stale_memory(0.1)
        
        # Get observability summary
        summary = self.dashboard.get_observability_summary()
        
        self.assertEqual(summary["health_status"], "HEALTHY")
        self.assertGreater(summary["cognitive_metrics"]["retrieval_hit_rate"], 0.0)
    
    def test_observability_with_explainability(self):
        """Test observability during explainability operations."""
        # Simulate explainability operations
        self.metrics_collector.record_explainability_consistency(0.9)
        self.metrics_collector.record_confidence(0.85)
        self.metrics_collector.record_contextual_confidence(0.82)
        
        # Validate explanation
        explanation = ContextualExplanation(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            timestamp=1234567890.0,
            current_regime="STARTUP",
            anomaly_score=0.85,
            primary_drivers=["Desviación Z-score (3.20σ)"],
            dynamic_context={"current_value": 85.2},
            similar_event_count=3,
            historical_context="3 eventos similares",
            historical_patterns=["STARTUP"],
            operational_confidence=0.82,
            suggested_actions=["Monitorear ramp-up"],
        )
        
        validation_result = self.explainability_validator.validate_explanation(
            explanation, retrieval_relevance=0.8
        )
        
        self.assertGreater(validation_result["explainability_quality_score"], 0.7)
    
    def test_observability_with_drift_detection(self):
        """Test observability during drift detection."""
        # Simulate drift
        current_regime = {"STARTUP": 0.7, "STABLE_NORMAL": 0.3}
        current_features = {"temperature": 75.0, "pressure": 100.0}
        current_anomaly = 0.2
        current_embedding = 0.5
        
        drift_result = self.drift_detection_engine.detect_drift(
            current_regime_distribution=current_regime,
            current_feature_means=current_features,
            current_anomaly_frequency=current_anomaly,
            current_embedding_mean=current_embedding,
        )
        
        self.assertTrue(drift_result.drift_detected)
        self.assertEqual(drift_result.drift_type, "regime")
    
    def test_observability_with_feedback_loop(self):
        """Test observability with feedback loop."""
        # Record feedback
        self.feedback_loop_manager.record_alert_feedback(sensor_id=12345, usefulness=0.8)
        self.feedback_loop_manager.record_retrieval_feedback(sensor_id=12345, usefulness=0.9)
        self.feedback_loop_manager.record_explainability_feedback(sensor_id=12345, usefulness=0.7)
        
        # Get feedback summary
        alert_summary = self.feedback_loop_manager.get_feedback_summary("alert_usefulness")
        retrieval_summary = self.feedback_loop_manager.get_feedback_summary("retrieval_usefulness")
        explainability_summary = self.feedback_loop_manager.get_feedback_summary("explainability_usefulness")
        
        self.assertEqual(alert_summary["count"], 1)
        self.assertEqual(retrieval_summary["count"], 1)
        self.assertEqual(explainability_summary["count"], 1)
    
    def test_observability_comprehensive_workflow(self):
        """Test comprehensive observability workflow."""
        # Simulate complete workflow
        self.metrics_collector.record_regime("STARTUP")
        self.metrics_collector.record_anomaly("ANOMALY_CONFIRMED")
        self.metrics_collector.record_retrieval(hit=True, similarity=0.85)
        self.metrics_collector.record_explainability_consistency(0.9)
        self.metrics_collector.record_confidence(0.85)
        
        self.memory_health_monitor.record_semantic_duplication(0.05)
        self.memory_health_monitor.record_stale_memory(0.1)
        
        self.feedback_loop_manager.record_alert_feedback(sensor_id=12345, usefulness=0.8)
        
        # Get comprehensive summary
        summary = self.dashboard.get_observability_summary()
        
        self.assertIn("cognitive_metrics", summary)
        self.assertIn("memory_health", summary)
        self.assertIn("feedback_summary", summary)
        self.assertIn("health_status", summary)
        
        self.assertEqual(summary["health_status"], "HEALTHY")
    
    def test_observability_health_status_calculation(self):
        """Test health status calculation with different scenarios."""
        # Healthy scenario
        self.metrics_collector.record_retrieval(hit=True, similarity=0.85)
        self.metrics_collector.record_explainability_consistency(0.9)
        self.memory_health_monitor.record_memory_explosion_risk(0.1)
        
        status = self.dashboard._calculate_health_status(
            self.metrics_collector.collect_metrics(),
            self.memory_health_monitor.assess_health(),
        )
        
        self.assertEqual(status, "HEALTHY")
        
        # Reset for warning scenario
        self.metrics_collector.reset()
        self.memory_health_monitor.reset()
        
        # Warning scenario
        self.metrics_collector.record_retrieval(hit=False)
        self.metrics_collector.record_explainability_consistency(0.6)
        self.memory_health_monitor.record_memory_explosion_risk(0.3)
        
        status = self.dashboard._calculate_health_status(
            self.metrics_collector.collect_metrics(),
            self.memory_health_monitor.assess_health(),
        )
        
        self.assertEqual(status, "WARNING")


if __name__ == "__main__":
    unittest.main()
