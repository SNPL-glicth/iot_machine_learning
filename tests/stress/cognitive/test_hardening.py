"""
Stress tests and hardening tests for cognitive components.

Tests for:
- Memory growth
- Drift simulation
- Retrieval degradation
- Explainability stability
- Operational stability
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import unittest
import time
from unittest.mock import Mock

from infrastructure.ml.cognitive.observability.cognitive_metrics_collector import CognitiveMetricsCollector
from infrastructure.ml.cognitive.observability.memory_health_monitor import MemoryHealthMonitor
from infrastructure.ml.cognitive.observability.drift_detection_engine import DriftDetectionEngine
from infrastructure.ml.cognitive.observability.explainability_validator import ExplainabilityValidator
from infrastructure.ml.cognitive.memory.semantic_event_builder import SemanticEventBuilder
from infrastructure.ml.cognitive.memory.operational_memory_pipeline import OperationalMemoryPipeline
from infrastructure.ml.cognitive.memory.cognitive_memory_registry import CognitiveMemoryRegistry
from domain.entities.memory import MemoryEvent
from domain.entities.explainability import ContextualExplanation


class TestCognitiveHardening(unittest.TestCase):
    """Stress tests and hardening tests for cognitive components."""
    
    def test_memory_growth_stress(self):
        """Test memory growth under stress."""
        collector = CognitiveMetricsCollector()
        
        # Simulate high volume of events
        for i in range(1000):
            collector.record_regime("STARTUP" if i % 2 == 0 else "STABLE_NORMAL")
            collector.record_anomaly("ANOMALY_CONFIRMED" if i % 3 == 0 else "ANOMALY_SUSPECTED")
            collector.record_retrieval(hit=(i % 4 != 0), similarity=0.8 + (i % 10) * 0.02)
        
        metrics = collector.collect_metrics()
        
        # Verify system can handle high volume
        self.assertEqual(metrics.regime_distribution["STARTUP"], 500)
        self.assertEqual(metrics.regime_distribution["STABLE_NORMAL"], 500)
        self.assertGreater(metrics.retrieval_hit_rate, 0.0)
    
    def test_drift_simulation(self):
        """Test drift detection under simulated drift conditions."""
        engine = DriftDetectionEngine()
        
        # Set baselines
        engine.set_baselines(
            regime_distribution={"STARTUP": 0.3, "STABLE_NORMAL": 0.7},
            feature_means={"temperature": 50.0, "pressure": 100.0},
            anomaly_frequency=0.1,
            embedding_mean=0.5,
        )
        
        # Simulate gradual drift
        for i in range(10):
            drift_factor = i * 0.05  # Gradual increase
            current_regime = {"STARTUP": 0.3 + drift_factor, "STABLE_NORMAL": 0.7 - drift_factor}
            current_features = {"temperature": 50.0 * (1 + drift_factor), "pressure": 100.0}
            
            result = engine.detect_drift(
                current_regime_distribution=current_regime,
                current_feature_means=current_features,
                current_anomaly_frequency=0.1,
                current_embedding_mean=0.5,
            )
            
            # Drift should become more detectable over time
            if i > 5:
                self.assertTrue(result.drift_detected)
    
    def test_retrieval_degradation_simulation(self):
        """Test retrieval degradation under stress."""
        monitor = MemoryHealthMonitor()
        
        # Simulate gradual retrieval degradation
        for i in range(10):
            degradation_score = i * 0.1  # Gradual degradation
            monitor.record_retrieval_degradation(degradation_score)
        
        health = monitor.assess_health()
        
        # Retrieval usefulness should decrease
        self.assertLess(health.retrieval_usefulness_score, 0.6)
        self.assertIn("degradation", health.cleanup_recommendations[0].lower())
    
    def test_explainability_stability(self):
        """Test explainability stability under stress."""
        validator = ExplainabilityValidator()
        
        # Generate multiple explanations for same sensor
        explanations = []
        for i in range(10):
            explanation = ContextualExplanation(
                sensor_id=12345,
                sensor_type="TEMPERATURE",
                timestamp=1234567890.0 + i,
                current_regime="STARTUP",
                anomaly_score=0.85 + i * 0.01,
                primary_drivers=["Desviación Z-score (3.20σ)", "Tasa de cambio (2.50)"],
                dynamic_context={"current_value": 85.2 + i},
                similar_event_count=3,
                historical_context="3 eventos similares",
                historical_patterns=["STARTUP"],
                operational_confidence=0.82,
                suggested_actions=["Monitorear ramp-up"],
            )
            explanations.append(explanation)
            validator.validate_explanation(explanation, retrieval_relevance=0.8)
        
        # Check for contradictions
        contradictions = validator.detect_contradictions(explanations)
        
        # Should not have regime contradictions
        regime_contradictions = [c for c in contradictions if c["type"] == "regime_contradiction"]
        self.assertEqual(len(regime_contradictions), 0)
    
    def test_operational_stability(self):
        """Test operational stability under stress."""
        collector = CognitiveMetricsCollector()
        monitor = MemoryHealthMonitor()
        
        # Simulate operational stress
        for i in range(100):
            collector.record_regime("STARTUP")
            collector.record_anomaly("ANOMALY_CONFIRMED")
            collector.record_retrieval(hit=True, similarity=0.85)
            collector.record_explainability_consistency(0.9)
            collector.record_confidence(0.85)
            
            monitor.record_semantic_duplication(0.05)
            monitor.record_stale_memory(0.1)
        
        # Verify system remains stable
        metrics = collector.collect_metrics()
        health = monitor.assess_health()
        
        self.assertGreater(metrics.explainability_consistency, 0.8)
        self.assertGreater(health.memory_quality_score, 0.7)
    
    def test_memory_explosion_prevention(self):
        """Test memory explosion prevention."""
        monitor = MemoryHealthMonitor()
        
        # Simulate high memory growth
        for i in range(10):
            explosion_risk = 0.8 + i * 0.02  # High risk
            monitor.record_memory_explosion_risk(explosion_risk)
        
        health = monitor.assess_health()
        
        # Should detect explosion risk
        self.assertGreater(health.memory_explosion_risk, 0.8)
        self.assertTrue(any("explosion" in rec.lower() for rec in health.cleanup_recommendations))
    
    def test_concurrent_operations(self):
        """Test system stability under concurrent operations."""
        collector = CognitiveMetricsCollector()
        
        # Simulate concurrent-like operations
        for i in range(500):
            collector.record_regime("STARTUP")
            collector.record_anomaly("ANOMALY_CONFIRMED")
            collector.record_retrieval(hit=True, similarity=0.85)
            collector.record_explainability_consistency(0.9)
            collector.record_confidence(0.85)
            collector.record_memory_growth(0.1)
            collector.record_ttl_cleanup(0.05)
            collector.record_contextual_confidence(0.82)
        
        metrics = collector.collect_metrics()
        
        # Verify system remains consistent
        self.assertEqual(metrics.regime_distribution["STARTUP"], 500)
        self.assertEqual(metrics.anomaly_distribution["ANOMALY_CONFIRMED"], 500)
        self.assertEqual(metrics.retrieval_hit_rate, 1.0)
    
    def test_latency_under_stress(self):
        """Test latency under stress conditions."""
        collector = CognitiveMetricsCollector()
        
        start_time = time.perf_counter()
        
        # Simulate high volume operations
        for i in range(1000):
            collector.record_regime("STARTUP")
            collector.record_anomaly("ANOMALY_CONFIRMED")
            collector.record_retrieval(hit=True, similarity=0.85)
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        # Latency should remain reasonable (< 100ms for 1000 operations)
        self.assertLess(latency_ms, 100.0)
    
    def test_reset_and_recovery(self):
        """Test system recovery after reset."""
        collector = CognitiveMetricsCollector()
        monitor = MemoryHealthMonitor()
        
        # Record metrics
        for i in range(100):
            collector.record_regime("STARTUP")
            monitor.record_semantic_duplication(0.1)
        
        # Reset
        collector.reset()
        monitor.reset()
        
        # Verify recovery
        metrics = collector.collect_metrics()
        health = monitor.assess_health()
        
        self.assertEqual(metrics.regime_distribution, {})
        self.assertEqual(health.semantic_duplication_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
