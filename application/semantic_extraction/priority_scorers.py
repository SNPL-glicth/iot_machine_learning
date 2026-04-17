"""Individual priority scoring strategies.

Each scorer implements a single scoring dimension.
Used by EntityPrioritizer for modular scoring.
"""

from __future__ import annotations

from typing import Any, Dict, List

from iot_machine_learning.domain.entities.semantic_extraction import (
    EntityType,
    SemanticEntity,
)


class TypeBasedScorer:
    """Scores entities based on semantic type importance."""
    name = "type_based"
    weight = 1.0
    
    TYPE_SCORES = {
        EntityType.ALERT: 150,
        EntityType.METRIC: 120,
        EntityType.EQUIPMENT: 100,
        EntityType.OPERATIONAL: 90,
        EntityType.COMPONENT: 70,
        EntityType.SYSTEM: 60,
        EntityType.LOCATION: 50,
        EntityType.TEMPORAL: 40,
        EntityType.PERSONNEL: 30,
    }
    
    def score(self, entity: SemanticEntity, context: Dict[str, Any]) -> float:
        return float(self.TYPE_SCORES.get(entity.entity_type, 50))


class MetricAnomalyScorer:
    """Scores metrics based on out-of-range severity."""
    name = "metric_anomaly"
    weight = 2.0  # High weight for anomalies
    
    def score(self, entity: SemanticEntity, context: Dict[str, Any]) -> float:
        if entity.entity_type != EntityType.METRIC:
            return 0.0
        
        attrs = entity.attributes
        if not attrs.get("is_out_of_range", False):
            return 0.0
        
        deviation = abs(attrs.get("deviation_percent", 0))
        return min(100, 50 + deviation * 0.5)


class ContextProximityScorer:
    """Scores entities based on proximity to urgency keywords."""
    name = "context_proximity"
    weight = 1.0
    
    URGENCY_KEYWORDS: List[str] = [
        "critical", "failure", "alarm", "emergency", "shutdown",
        "trip", "overheat", "overpressure", "leak", "rupture",
        "anomaly", "deviation", "fault", "error", "warning",
        "high temperature", "high pressure", "low flow",
    ]
    
    def score(self, entity: SemanticEntity, context: Dict[str, Any]) -> float:
        if not entity.context_window:
            return 0.0
        
        window_lower = entity.context_window.lower()
        matches = sum(1 for kw in self.URGENCY_KEYWORDS if kw in window_lower)
        
        return min(60, matches * 12)  # 12 per match, max 60


class DocumentPositionScorer:
    """Scores entities based on document position importance."""
    name = "document_position"
    weight = 0.8
    
    POSITION_SCORES = {
        "title": 40,
        "opening": 30,
        "conclusion": 20,
        "body": 5,
    }
    
    def score(self, entity: SemanticEntity, context: Dict[str, Any]) -> float:
        position = context.get("document_position", "body")
        return float(self.POSITION_SCORES.get(position, 0))


class EquipmentCriticalPathScorer:
    """Scores equipment on critical path with higher priority."""
    name = "critical_path"
    weight = 1.5
    
    def score(self, entity: SemanticEntity, context: Dict[str, Any]) -> float:
        if entity.entity_type != EntityType.EQUIPMENT:
            return 0.0
        
        attrs = entity.attributes
        if attrs.get("is_critical_path", False):
            return 50.0
        
        # Check if parent system is critical
        parent = attrs.get("parent_system", "")
        critical_systems = ["compressor", "turbine", "reactor", "boiler"]
        if any(sys in parent.lower() for sys in critical_systems):
            return 30.0
        
        return 0.0


class RelationDensityScorer:
    """Scores entities based on number of relations (connectedness)."""
    name = "relation_density"
    weight = 0.6
    
    def score(self, entity: SemanticEntity, context: Dict[str, Any]) -> float:
        n_relations = len(entity.relations)
        # More relations = more central to the narrative
        return min(30, n_relations * 10)
