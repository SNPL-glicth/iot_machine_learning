"""
Unit tests for OperationalDependencyGraphManager.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import unittest

from infrastructure.ml.cognitive.causal.operational_dependency_graph_manager import OperationalDependencyGraphManager
from domain.entities.causal import CausalCorrelation


class TestOperationalDependencyGraphManager(unittest.TestCase):
    """Test cases for OperationalDependencyGraphManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = OperationalDependencyGraphManager()
    
    def test_add_correlation(self):
        """Test adding correlation to graph."""
        correlation = CausalCorrelation(
            source_sensor_id=12345,
            target_sensor_id=67890,
            correlation_coefficient=0.8,
            lag_seconds=10.0,
            confidence=0.85,
            propagation_likelihood=0.75,
            timestamp=1234567890.0,
        )
        
        self.manager.add_correlation(correlation)
        
        # Nodes should be added
        self.assertIn(12345, self.manager._nodes)
        self.assertIn(67890, self.manager._nodes)
        
        # Edge should be added
        self.assertEqual(len(self.manager._edges), 1)
    
    def test_add_correlation_duplicate(self):
        """Test adding duplicate correlation."""
        correlation1 = CausalCorrelation(
            source_sensor_id=12345,
            target_sensor_id=67890,
            correlation_coefficient=0.8,
            lag_seconds=10.0,
            confidence=0.85,
            propagation_likelihood=0.75,
            timestamp=1234567890.0,
        )
        
        correlation2 = CausalCorrelation(
            source_sensor_id=12345,
            target_sensor_id=67890,
            correlation_coefficient=0.7,
            lag_seconds=15.0,
            confidence=0.75,
            propagation_likelihood=0.65,
            timestamp=1234567891.0,
        )
        
        self.manager.add_correlation(correlation1)
        self.manager.add_correlation(correlation2)
        
        # Should have one edge with averaged values
        self.assertEqual(len(self.manager._edges), 1)
        edge = self.manager._edges[0]
        self.assertAlmostEqual(edge.weight, 0.7, places=2)  # (0.75 + 0.65) / 2
        self.assertAlmostEqual(edge.confidence, 0.8, places=2)  # (0.85 + 0.75) / 2
    
    def test_build_graph(self):
        """Test building dependency graph."""
        correlation = CausalCorrelation(
            source_sensor_id=12345,
            target_sensor_id=67890,
            correlation_coefficient=0.8,
            lag_seconds=10.0,
            confidence=0.85,
            propagation_likelihood=0.75,
            timestamp=1234567890.0,
        )
        
        self.manager.add_correlation(correlation)
        graph = self.manager.build_graph()
        
        self.assertEqual(len(graph.nodes), 2)
        self.assertEqual(len(graph.edges), 1)
    
    def test_get_propagation_path(self):
        """Test getting propagation path."""
        # Add correlations to create a path: 12345 -> 67890 -> 11111
        self.manager.add_correlation(CausalCorrelation(
            source_sensor_id=12345,
            target_sensor_id=67890,
            correlation_coefficient=0.8,
            lag_seconds=10.0,
            confidence=0.85,
            propagation_likelihood=0.75,
            timestamp=1234567890.0,
        ))
        
        self.manager.add_correlation(CausalCorrelation(
            source_sensor_id=67890,
            target_sensor_id=11111,
            correlation_coefficient=0.8,
            lag_seconds=10.0,
            confidence=0.85,
            propagation_likelihood=0.75,
            timestamp=1234567890.0,
        ))
        
        path = self.manager.get_propagation_path(12345, 11111)
        
        self.assertIsNotNone(path)
        self.assertEqual(path, [12345, 67890, 11111])
    
    def test_get_propagation_path_no_path(self):
        """Test getting propagation path when no path exists."""
        path = self.manager.get_propagation_path(12345, 67890)
        
        self.assertIsNone(path)
    
    def test_get_neighbors(self):
        """Test getting neighboring sensors."""
        self.manager.add_correlation(CausalCorrelation(
            source_sensor_id=12345,
            target_sensor_id=67890,
            correlation_coefficient=0.8,
            lag_seconds=10.0,
            confidence=0.85,
            propagation_likelihood=0.75,
            timestamp=1234567890.0,
        ))
        
        neighbors = self.manager.get_neighbors(12345)
        
        self.assertEqual(neighbors, [67890])
    
    def test_apply_temporal_decay(self):
        """Test applying temporal decay."""
        correlation = CausalCorrelation(
            source_sensor_id=12345,
            target_sensor_id=67890,
            correlation_coefficient=0.8,
            lag_seconds=10.0,
            confidence=0.85,
            propagation_likelihood=0.5,  # Low weight
            timestamp=1234567890.0,
        )
        
        self.manager.add_correlation(correlation)
        
        # Manually age the edge
        edge_key = (12345, 67890)
        self.manager._edge_timestamps[edge_key] = 1234567890.0 - 3600.0  # 1 hour ago
        
        self.manager.apply_temporal_decay()
        
        # Edge should be removed due to decay
        self.assertEqual(len(self.manager._edges), 0)
    
    def test_reset(self):
        """Test resetting graph manager."""
        correlation = CausalCorrelation(
            source_sensor_id=12345,
            target_sensor_id=67890,
            correlation_coefficient=0.8,
            lag_seconds=10.0,
            confidence=0.85,
            propagation_likelihood=0.75,
            timestamp=1234567890.0,
        )
        
        self.manager.add_correlation(correlation)
        self.manager.reset()
        
        self.assertEqual(len(self.manager._nodes), 0)
        self.assertEqual(len(self.manager._edges), 0)


if __name__ == "__main__":
    unittest.main()
