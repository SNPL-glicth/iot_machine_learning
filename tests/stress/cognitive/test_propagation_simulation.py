"""
Propagation simulation tests for causal mapping components.

Tests for:
- Propagation simulation
- Temporal consistency
- Graph stability
- False causality detection
- Operational stability
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import unittest
import time

from infrastructure.ml.cognitive.causal.causal_correlation_engine import CausalCorrelationEngine
from infrastructure.ml.cognitive.causal.operational_dependency_graph_manager import OperationalDependencyGraphManager
from infrastructure.ml.cognitive.causal.temporal_pattern_miner import TemporalPatternMiner
from infrastructure.ml.cognitive.causal.event_propagation_tracker import EventPropagationTracker
from infrastructure.ml.cognitive.causal.propagation_confidence_calculator import PropagationConfidenceCalculator
from infrastructure.ml.cognitive.causal.operational_sequence_registry import OperationalSequenceRegistry
from domain.entities.causal import CausalCorrelation


class TestPropagationSimulation(unittest.TestCase):
    """Propagation simulation tests for causal mapping."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.correlation_engine = CausalCorrelationEngine()
        self.graph_manager = OperationalDependencyGraphManager()
        self.pattern_miner = TemporalPatternMiner()
        self.propagation_tracker = EventPropagationTracker()
        self.confidence_calculator = PropagationConfidenceCalculator()
        self.sequence_registry = OperationalSequenceRegistry()
    
    def test_propagation_simulation_linear(self):
        """Test linear propagation simulation."""
        # Simulate linear propagation: 12345 -> 67890 -> 11111
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
        
        # Detect correlations
        correlations = self.correlation_engine.detect_correlations(12345)
        
        # Should detect linear propagation
        self.assertGreater(len(correlations), 0)
        
        # Build graph
        for correlation in correlations:
            self.graph_manager.add_correlation(correlation)
        
        graph = self.graph_manager.build_graph()
        
        # Should have linear chain
        path = self.graph_manager.get_propagation_path(12345, 11111)
        self.assertIsNotNone(path)
    
    def test_propagation_simulation_cascade(self):
        """Test cascade propagation simulation."""
        # Simulate cascade: 12345 -> [67890, 11111, 22222]
        for i in range(20):
            self.correlation_engine.add_sensor_reading(
                sensor_id=12345,
                value=50.0 + i,
                timestamp=1234567890.0 + i,
            )
            for target_id in [67890, 11111, 22222]:
                self.correlation_engine.add_sensor_reading(
                    sensor_id=target_id,
                    value=50.0 + i * 0.8,
                    timestamp=1234567890.0 + i + 5,
                )
        
        # Detect correlations
        correlations = self.correlation_engine.detect_correlations(12345)
        
        # Should detect multiple targets
        self.assertGreater(len(correlations), 1)
        
        # Track cascade
        propagation_id = self.propagation_tracker.start_propagation(
            source_sensor_id=12345,
            timestamp=1234567890.0,
        )
        
        for target_id in [67890, 11111, 22222]:
            self.propagation_tracker.add_to_propagation(
                propagation_id=propagation_id,
                target_sensor_id=target_id,
                timestamp=1234567891.0,
            )
        
        event = self.propagation_tracker.end_propagation(
            propagation_id=propagation_id,
            end_timestamp=1234567892.0,
        )
        
        # Should be cascade
        self.assertTrue(event.is_cascade)
    
    def test_temporal_consistency(self):
        """Test temporal consistency of correlations."""
        # Add data with consistent lag
        for i in range(20):
            self.correlation_engine.add_sensor_reading(
                sensor_id=12345,
                value=50.0 + i,
                timestamp=1234567890.0 + i,
            )
            self.correlation_engine.add_sensor_reading(
                sensor_id=67890,
                value=50.0 + i * 0.9,
                timestamp=1234567890.0 + i + 10,  # Consistent 10s lag
            )
        
        correlation1 = self.correlation_engine._compute_lagged_correlation(12345, 67890)
        
        # Add more data with same lag
        for i in range(20, 40):
            self.correlation_engine.add_sensor_reading(
                sensor_id=12345,
                value=50.0 + i,
                timestamp=1234567890.0 + i,
            )
            self.correlation_engine.add_sensor_reading(
                sensor_id=67890,
                value=50.0 + i * 0.9,
                timestamp=1234567890.0 + i + 10,
            )
        
        correlation2 = self.correlation_engine._compute_lagged_correlation(12345, 67890)
        
        # Correlations should be consistent
        self.assertAlmostEqual(
            correlation1.correlation_coefficient,
            correlation2.correlation_coefficient,
            places=1,
        )
    
    def test_graph_stability(self):
        """Test graph stability over time."""
        # Add initial correlations
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
        
        graph2 = self.graph_manager.build_graph()
        
        # Graph should be stable (nodes should remain consistent)
        self.assertEqual(graph1.nodes, graph2.nodes)
        self.assertGreater(len(graph2.edges), len(graph1.edges))
    
    def test_false_causality_detection(self):
        """Test detection of false causality (spurious correlations)."""
        # Add uncorrelated data
        for i in range(20):
            self.correlation_engine.add_sensor_reading(
                sensor_id=12345,
                value=50.0 + i,
                timestamp=1234567890.0 + i,
            )
            self.correlation_engine.add_sensor_reading(
                sensor_id=67890,
                value=50.0 + (i % 5),  # Random pattern
                timestamp=1234567890.0 + i,
            )
        
        correlations = self.correlation_engine.detect_correlations(12345, [67890])
        
        # Should not detect strong correlation
        self.assertEqual(len(correlations), 0)
    
    def test_operational_stability(self):
        """Test operational stability under stress."""
        # Simulate high volume of sensor data
        for i in range(100):
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
        start_time = time.perf_counter()
        correlations = self.correlation_engine.detect_correlations(12345, [67890])
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        
        # Latency should be reasonable (< 100ms)
        self.assertLess(latency_ms, 100.0)
    
    def test_pattern_stability(self):
        """Test pattern stability over time."""
        # Add consistent pattern
        for i in range(10):
            sensor_sequence = [12345, 67890, 11111]
            timestamps = [1234567890.0 + i * 10, 1234567891.0 + i * 10, 1234567892.0 + i * 10]
            self.pattern_miner.add_event_sequence(sensor_sequence, timestamps)
        
        patterns1 = self.pattern_miner.mine_patterns()
        
        # Add more of same pattern
        for i in range(10, 20):
            sensor_sequence = [12345, 67890, 11111]
            timestamps = [1234567890.0 + i * 10, 1234567891.0 + i * 10, 1234567892.0 + i * 10]
            self.pattern_miner.add_event_sequence(sensor_sequence, timestamps)
        
        patterns2 = self.pattern_miner.mine_patterns()
        
        # Patterns should be stable
        self.assertEqual(len(patterns1), len(patterns2))
    
    def test_confidence_calibration(self):
        """Test confidence calibration."""
        # High frequency, high consistency
        confidence1 = self.confidence_calculator.calculate(
            historical_frequency=0.9,
            temporal_consistency=0.95,
            contextual_stability=0.85,
            operational_correlation=0.9,
        )
        
        # Low frequency, low consistency
        confidence2 = self.confidence_calculator.calculate(
            historical_frequency=0.2,
            temporal_consistency=0.3,
            contextual_stability=0.4,
            operational_correlation=0.5,
        )
        
        # High confidence should be greater than low confidence
        self.assertGreater(confidence1, confidence2)
    
    def test_sequence_registry_stability(self):
        """Test sequence registry stability."""
        # Register patterns
        for i in range(10):
            from domain.entities.causal import TemporalPattern
            pattern = TemporalPattern(
                pattern_id=f"pattern_{i}",
                sequence=[12345, 67890, 11111],
                frequency=5 + i,
                avg_duration_seconds=10.0,
                confidence=0.85,
                is_pre_anomaly=False,
                timestamp=1234567890.0 + i,
            )
            self.sequence_registry.register_sequence(pattern)
        
        stats1 = self.sequence_registry.get_sequence_statistics()
        
        # Add more patterns
        for i in range(10, 20):
            from domain.entities.causal import TemporalPattern
            pattern = TemporalPattern(
                pattern_id=f"pattern_{i}",
                sequence=[12345, 67890, 11111],
                frequency=5 + i,
                avg_duration_seconds=10.0,
                confidence=0.85,
                is_pre_anomaly=False,
                timestamp=1234567890.0 + i,
            )
            self.sequence_registry.register_sequence(pattern)
        
        stats2 = self.sequence_registry.get_sequence_statistics()
        
        # Registry should be stable
        self.assertGreater(stats2["total_sequences"], stats1["total_sequences"])


if __name__ == "__main__":
    unittest.main()
