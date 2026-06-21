"""
Integration tests for causal mapping with memory and observability.

These tests integrate causal mapping components with memory and observability systems.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import unittest

from infrastructure.ml.cognitive.causal.causal_correlation_engine import CausalCorrelationEngine
from infrastructure.ml.cognitive.causal.operational_dependency_graph_manager import OperationalDependencyGraphManager
from infrastructure.ml.cognitive.causal.temporal_pattern_miner import TemporalPatternMiner
from infrastructure.ml.cognitive.causal.event_propagation_tracker import EventPropagationTracker
from infrastructure.ml.cognitive.causal.propagation_confidence_calculator import PropagationConfidenceCalculator
from infrastructure.ml.cognitive.causal.operational_sequence_registry import OperationalSequenceRegistry
from infrastructure.ml.cognitive.observability.cognitive_metrics_collector import CognitiveMetricsCollector
from infrastructure.ml.cognitive.memory.semantic_event_builder import SemanticEventBuilder
from domain.entities.causal import CausalCorrelation, TemporalPattern


class TestCausalMappingIntegration(unittest.TestCase):
    """Integration tests for causal mapping."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.correlation_engine = CausalCorrelationEngine()
        self.graph_manager = OperationalDependencyGraphManager()
        self.pattern_miner = TemporalPatternMiner()
        self.propagation_tracker = EventPropagationTracker()
        self.confidence_calculator = PropagationConfidenceCalculator()
        self.sequence_registry = OperationalSequenceRegistry()
        self.metrics_collector = CognitiveMetricsCollector()
        self.event_builder = SemanticEventBuilder()
    
    def test_correlation_to_graph_integration(self):
        """Test integration between correlation engine and graph manager."""
        # Add sensor data
        for i in range(20):
            self.correlation_engine.add_sensor_reading(
                sensor_id=12345,
                value=50.0 + i,
                timestamp=1234567890.0 + i,
            )
            self.correlation_engine.add_sensor_reading(
                sensor_id=67890,
                value=50.0 + i * 0.9,
                timestamp=1234567890.0 + i + 5,
            )
        
        # Detect correlations
        correlations = self.correlation_engine.detect_correlations(12345, [67890])
        
        # Add correlations to graph
        for correlation in correlations:
            self.graph_manager.add_correlation(correlation)
        
        # Build graph
        graph = self.graph_manager.build_graph()
        
        self.assertGreater(len(graph.nodes), 0)
        self.assertGreater(len(graph.edges), 0)
    
    def test_pattern_to_registry_integration(self):
        """Test integration between pattern miner and sequence registry."""
        # Add event sequences
        for i in range(5):
            sensor_sequence = [12345, 67890, 11111]
            timestamps = [1234567890.0 + i * 10, 1234567891.0 + i * 10, 1234567892.0 + i * 10]
            self.pattern_miner.add_event_sequence(sensor_sequence, timestamps)
        
        # Mine patterns
        patterns = self.pattern_miner.mine_patterns()
        
        # Register patterns
        for pattern in patterns:
            self.sequence_registry.register_sequence(pattern)
        
        # Get frequent sequences
        frequent = self.sequence_registry.get_frequent_sequences(min_frequency=3)
        
        self.assertGreater(len(frequent), 0)
    
    def test_propagation_with_confidence_integration(self):
        """Test integration between propagation tracker and confidence calculator."""
        # Start propagation
        propagation_id = self.propagation_tracker.start_propagation(
            source_sensor_id=12345,
            timestamp=1234567890.0,
        )
        
        # Add targets
        self.propagation_tracker.add_to_propagation(
            propagation_id=propagation_id,
            target_sensor_id=67890,
            timestamp=1234567891.0,
        )
        
        self.propagation_tracker.add_to_propagation(
            propagation_id=propagation_id,
            target_sensor_id=11111,
            timestamp=1234567892.0,
        )
        
        # End propagation
        event = self.propagation_tracker.end_propagation(
            propagation_id=propagation_id,
            end_timestamp=1234567893.0,
        )
        
        # Calculate confidence
        confidence = self.confidence_calculator.calculate(
            historical_frequency=0.8,
            temporal_consistency=0.9,
            contextual_stability=0.7,
            operational_correlation=0.85,
        )
        
        self.assertIsNotNone(event)
        self.assertGreater(confidence, 0.7)
    
    def test_causal_with_observability_integration(self):
        """Test integration between causal mapping and observability."""
        # Record causal metrics
        self.metrics_collector.record_regime("STARTUP")
        self.metrics_collector.record_anomaly("ANOMALY_CONFIRMED")
        
        # Detect correlations
        for i in range(20):
            self.correlation_engine.add_sensor_reading(
                sensor_id=12345,
                value=50.0 + i,
                timestamp=1234567890.0 + i,
            )
            self.correlation_engine.add_sensor_reading(
                sensor_id=67890,
                value=50.0 + i * 0.9,
                timestamp=1234567890.0 + i + 5,
            )
        
        correlations = self.correlation_engine.detect_correlations(12345, [67890])
        
        # Record retrieval metrics
        for correlation in correlations:
            self.metrics_collector.record_retrieval(
                hit=True,
                similarity=correlation.correlation_coefficient,
            )
        
        # Get metrics
        metrics = self.metrics_collector.collect_metrics()
        
        self.assertGreater(metrics.retrieval_hit_rate, 0.0)
    
    def test_comprehensive_causal_workflow(self):
        """Test comprehensive causal mapping workflow."""
        # 1. Add sensor data
        for i in range(20):
            self.correlation_engine.add_sensor_reading(
                sensor_id=12345,
                value=50.0 + i,
                timestamp=1234567890.0 + i,
            )
            self.correlation_engine.add_sensor_reading(
                sensor_id=67890,
                value=50.0 + i * 0.9,
                timestamp=1234567890.0 + i + 5,
            )
            self.correlation_engine.add_sensor_reading(
                sensor_id=11111,
                value=50.0 + i * 0.8,
                timestamp=1234567890.0 + i + 10,
            )
        
        # 2. Detect correlations
        correlations = self.correlation_engine.detect_correlations(12345)
        
        # 3. Build dependency graph
        for correlation in correlations:
            self.graph_manager.add_correlation(correlation)
        
        graph = self.graph_manager.build_graph()
        
        # 4. Track propagation
        propagation_id = self.propagation_tracker.start_propagation(
            source_sensor_id=12345,
            timestamp=1234567890.0,
        )
        
        self.propagation_tracker.add_to_propagation(
            propagation_id=propagation_id,
            target_sensor_id=67890,
            timestamp=1234567891.0,
        )
        
        self.propagation_tracker.end_propagation(
            propagation_id=propagation_id,
            end_timestamp=1234567892.0,
        )
        
        # 5. Mine patterns
        for i in range(5):
            sensor_sequence = [12345, 67890, 11111]
            timestamps = [1234567890.0 + i * 10, 1234567891.0 + i * 10, 1234567892.0 + i * 10]
            self.pattern_miner.add_event_sequence(sensor_sequence, timestamps)
        
        patterns = self.pattern_miner.mine_patterns()
        
        # Verify workflow completed
        self.assertGreater(len(graph.nodes), 0)
        self.assertGreater(len(self.propagation_tracker.get_completed_propagations()), 0)
        self.assertGreater(len(patterns), 0)
    
    def test_graph_stability_over_time(self):
        """Test graph stability over time."""
        # Add correlations
        for i in range(10):
            correlation = CausalCorrelation(
                source_sensor_id=12345,
                target_sensor_id=67890,
                correlation_coefficient=0.8,
                lag_seconds=10.0,
                confidence=0.85,
                propagation_likelihood=0.75,
                timestamp=1234567890.0 + i,
            )
            self.graph_manager.add_correlation(correlation)
        
        # Build graph
        graph1 = self.graph_manager.build_graph()
        
        # Add more correlations
        for i in range(10):
            correlation = CausalCorrelation(
                source_sensor_id=12345,
                target_sensor_id=11111,
                correlation_coefficient=0.7,
                lag_seconds=15.0,
                confidence=0.75,
                propagation_likelihood=0.65,
                timestamp=1234567890.0 + i,
            )
            self.graph_manager.add_correlation(correlation)
        
        # Build graph again
        graph2 = self.graph_manager.build_graph()
        
        # Graph should be stable (nodes should remain consistent)
        self.assertIn(12345, graph1.nodes)
        self.assertIn(12345, graph2.nodes)


if __name__ == "__main__":
    unittest.main()
