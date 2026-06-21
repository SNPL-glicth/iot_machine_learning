"""
OperationalDependencyGraph domain entity for operational dependency modeling.

This is a domain entity for representing operational dependency graphs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass(frozen=True)
class DependencyEdge:
    """Edge in operational dependency graph."""
    source_sensor_id: int
    target_sensor_id: int
    weight: float
    confidence: float
    temporal_decay: float


@dataclass(frozen=True)
class OperationalDependencyGraph:
    """
    Domain entity for operational dependency graph (DDD value object).
    
    This represents the domain concept of operational dependency graph.
    """
    
    timestamp: float
    nodes: List[int]  # sensor IDs
    edges: List[DependencyEdge]
    graph_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "nodes": self.nodes,
            "edges": [
                {
                    "source_sensor_id": edge.source_sensor_id,
                    "target_sensor_id": edge.target_sensor_id,
                    "weight": edge.weight,
                    "confidence": edge.confidence,
                    "temporal_decay": edge.temporal_decay,
                }
                for edge in self.edges
            ],
            "graph_metadata": self.graph_metadata,
        }
