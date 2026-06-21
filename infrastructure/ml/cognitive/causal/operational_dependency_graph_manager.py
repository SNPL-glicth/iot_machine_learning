"""
OperationalDependencyGraphManager for modeling operational dependency graphs.

Implements nodes by sensor/equipment, weighted edges by propagation, confidence by edge, simple temporal decay, and dynamic updates.
"""

from typing import List, Dict, Any, Optional
import time

from domain.entities.causal import OperationalDependencyGraph, DependencyEdge, CausalCorrelation


class OperationalDependencyGraphManager:
    """Manager for operational dependency graphs."""
    
    def __init__(
        self,
        min_edge_weight: float = 0.3,
        min_edge_confidence: float = 0.5,
        temporal_decay_rate: float = 0.1,
    ):
        """
        Initialize dependency graph manager.
        
        Args:
            min_edge_weight: Minimum edge weight threshold
            min_edge_confidence: Minimum edge confidence threshold
            temporal_decay_rate: Rate of temporal decay per hour
        """
        self._min_edge_weight = min_edge_weight
        self._min_edge_confidence = min_edge_confidence
        self._temporal_decay_rate = temporal_decay_rate
        
        # Current graph state
        self._nodes: List[int] = []
        self._edges: List[DependencyEdge] = []
        self._edge_timestamps: Dict[tuple, float] = {}
    
    def add_correlation(self, correlation: CausalCorrelation) -> None:
        """
        Add correlation to dependency graph.
        
        Args:
            correlation: Causal correlation to add
        """
        # Add nodes if not present
        if correlation.source_sensor_id not in self._nodes:
            self._nodes.append(correlation.source_sensor_id)
        if correlation.target_sensor_id not in self._nodes:
            self._nodes.append(correlation.target_sensor_id)
        
        # Create edge
        edge = DependencyEdge(
            source_sensor_id=correlation.source_sensor_id,
            target_sensor_id=correlation.target_sensor_id,
            weight=correlation.propagation_likelihood,
            confidence=correlation.confidence,
            temporal_decay=0.0,
        )
        
        # Check if edge already exists
        edge_key = (edge.source_sensor_id, edge.target_sensor_id)
        if edge_key in self._edge_timestamps:
            # Update existing edge
            existing_edge = next(
                e for e in self._edges
                if e.source_sensor_id == edge.source_sensor_id
                and e.target_sensor_id == edge.target_sensor_id
            )
            
            # Update edge with weighted average
            new_weight = (existing_edge.weight + edge.weight) / 2
            new_confidence = (existing_edge.confidence + edge.confidence) / 2
            
            # Replace edge
            self._edges = [
                e for e in self._edges
                if not (e.source_sensor_id == edge.source_sensor_id
                       and e.target_sensor_id == edge.target_sensor_id)
            ]
            
            edge = DependencyEdge(
                source_sensor_id=edge.source_sensor_id,
                target_sensor_id=edge.target_sensor_id,
                weight=new_weight,
                confidence=new_confidence,
                temporal_decay=0.0,
            )
        
        self._edges.append(edge)
        self._edge_timestamps[edge_key] = time.time()
    
    def apply_temporal_decay(self) -> None:
        """Apply temporal decay to edges."""
        current_time = time.time()
        
        decayed_edges = []
        for edge in self._edges:
            edge_key = (edge.source_sensor_id, edge.target_sensor_id)
            edge_age_hours = (current_time - self._edge_timestamps[edge_key]) / 3600.0
            
            # Calculate decay
            decay = 1.0 - (self._temporal_decay_rate * edge_age_hours)
            decay = max(0.0, decay)
            
            # Apply decay to weight
            decayed_weight = edge.weight * decay
            
            # Remove edge if weight is too low
            if decayed_weight >= self._min_edge_weight:
                decayed_edge = DependencyEdge(
                    source_sensor_id=edge.source_sensor_id,
                    target_sensor_id=edge.target_sensor_id,
                    weight=decayed_weight,
                    confidence=edge.confidence,
                    temporal_decay=decay,
                )
                decayed_edges.append(decayed_edge)
        
        self._edges = decayed_edges
    
    def build_graph(self) -> OperationalDependencyGraph:
        """Build current dependency graph."""
        # Apply temporal decay before building
        self.apply_temporal_decay()
        
        return OperationalDependencyGraph(
            timestamp=time.time(),
            nodes=self._nodes.copy(),
            edges=self._edges.copy(),
            graph_metadata={
                "node_count": len(self._nodes),
                "edge_count": len(self._edges),
                "min_edge_weight": self._min_edge_weight,
                "min_edge_confidence": self._min_edge_confidence,
            },
        )
    
    def get_propagation_path(
        self,
        source_sensor_id: int,
        target_sensor_id: int,
        max_depth: int = 3,
    ) -> Optional[List[int]]:
        """
        Get propagation path between sensors.
        
        Args:
            source_sensor_id: Source sensor ID
            target_sensor_id: Target sensor ID
            max_depth: Maximum path depth
        
        Returns:
            Propagation path or None
        """
        # BFS to find path
        from collections import deque
        
        queue = deque([(source_sensor_id, [source_sensor_id])])
        visited = {source_sensor_id}
        
        while queue:
            current, path = queue.popleft()
            
            if current == target_sensor_id:
                return path
            
            if len(path) >= max_depth:
                continue
            
            # Find outgoing edges
            outgoing_edges = [
                e for e in self._edges
                if e.source_sensor_id == current
                and e.weight >= self._min_edge_weight
                and e.confidence >= self._min_edge_confidence
            ]
            
            for edge in outgoing_edges:
                if edge.target_sensor_id not in visited:
                    visited.add(edge.target_sensor_id)
                    queue.append((edge.target_sensor_id, path + [edge.target_sensor_id]))
        
        return None
    
    def get_neighbors(self, sensor_id: int) -> List[int]:
        """
        Get neighboring sensors.
        
        Args:
            sensor_id: Sensor ID
        
        Returns:
            List of neighboring sensor IDs
        """
        neighbors = []
        
        for edge in self._edges:
            if edge.source_sensor_id == sensor_id:
                neighbors.append(edge.target_sensor_id)
        
        return neighbors
    
    def reset(self) -> None:
        """Reset graph state."""
        self._nodes = []
        self._edges = []
        self._edge_timestamps = {}
