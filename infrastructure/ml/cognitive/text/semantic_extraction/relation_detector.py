"""RelationDetector — detect semantic relations between entities.

Uses proximity and contextual patterns to establish relationships.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from iot_machine_learning.domain.entities.semantic_extraction import (
    EntityRelation,
    EntityType,
    RelationType,
    SemanticEntity,
)


class RelationDetector:
    """Detect relations between extracted entities.
    
    Primary focus: Equipment-Metric relationships (C-12 → 3401 PSI)
    """
    
    # Maximum character distance for relation
    PROXIMITY_WINDOW = 80
    
    # Relation indicator keywords
    RELATION_INDICATORS = {
        RelationType.HAS_METRIC: [
            'reading', 'shows', 'indicates', 'at', 'pressure', 'temperature',
            'flow', 'level', 'measured', 'value', 'is', 'was', 'reading of',
        ],
        RelationType.LOCATED_AT: [
            'located', 'installed', 'at', 'in', 'position', 'area', 'sector',
        ],
        RelationType.CAUSED_BY: [
            'caused', 'due to', 'because', 'result of', 'triggered by',
        ],
        RelationType.TRIGGERS: [
            'triggers', 'causes', 'leads to', 'results in', 'activates',
        ],
    }
    
    def detect_relations(
        self,
        entities: List[SemanticEntity],
        text: str,
    ) -> List[EntityRelation]:
        """Detect relations between entities in text."""
        relations = []
        
        if len(entities) < 2:
            return relations
        
        # Index entities by position for efficient lookup
        entity_positions = [(e.start_pos, e.end_pos, i) for i, e in enumerate(entities)]
        entity_positions.sort()
        
        # Primary: Equipment-Metric relations
        equipment_indices = [
            i for i, e in enumerate(entities) 
            if e.entity_type == EntityType.EQUIPMENT
        ]
        metric_indices = [
            i for i, e in enumerate(entities) 
            if e.entity_type == EntityType.METRIC
        ]
        
        # Detect equipment-metric pairs by proximity
        for eq_idx in equipment_indices:
            eq_entity = entities[eq_idx]
            
            for met_idx in metric_indices:
                met_entity = entities[met_idx]
                
                # Calculate distance
                distance = abs(eq_entity.end_pos - met_entity.start_pos)
                if distance > self.PROXIMITY_WINDOW:
                    continue
                
                # Check for relation indicators in context
                context = self._extract_context(text, eq_entity.end_pos, met_entity.start_pos)
                confidence = self._compute_relation_confidence(
                    RelationType.HAS_METRIC, context, distance
                )
                
                if confidence > 0.5:
                    relation = EntityRelation(
                        source_idx=eq_idx,
                        target_idx=met_idx,
                        relation_type=RelationType.HAS_METRIC,
                        confidence=confidence,
                        context=context[:50],
                    )
                    relations.append(relation)
        
        # Detect equipment-location relations
        location_indices = [
            i for i, e in enumerate(entities)
            if e.entity_type == EntityType.LOCATION
        ]
        
        for eq_idx in equipment_indices:
            eq_entity = entities[eq_idx]
            
            for loc_idx in location_indices:
                loc_entity = entities[loc_idx]
                distance = abs(eq_entity.end_pos - loc_entity.start_pos)
                
                if distance <= self.PROXIMITY_WINDOW:
                    context = self._extract_context(text, eq_entity.end_pos, loc_entity.start_pos)
                    confidence = self._compute_relation_confidence(
                        RelationType.LOCATED_AT, context, distance
                    )
                    
                    if confidence > 0.5:
                        relation = EntityRelation(
                            source_idx=eq_idx,
                            target_idx=loc_idx,
                            relation_type=RelationType.LOCATED_AT,
                            confidence=confidence,
                            context=context[:50],
                        )
                        relations.append(relation)
        
        return relations
    
    def _extract_context(self, text: str, pos1: int, pos2: int) -> str:
        """Extract text between two positions."""
        start = min(pos1, pos2)
        end = max(pos1, pos2)
        return text[start:end].strip()
    
    def _compute_relation_confidence(
        self,
        rel_type: RelationType,
        context: str,
        distance: int,
    ) -> float:
        """Compute confidence for a potential relation."""
        context_lower = context.lower()
        
        # Base confidence from proximity (closer = higher)
        proximity_score = max(0, 1.0 - (distance / self.PROXIMITY_WINDOW))
        
        # Boost for indicator keywords
        indicators = self.RELATION_INDICATORS.get(rel_type, [])
        keyword_matches = sum(1 for ind in indicators if ind in context_lower)
        keyword_score = min(0.4, keyword_matches * 0.15)
        
        # Combine scores
        confidence = 0.4 + (proximity_score * 0.4) + keyword_score
        
        return min(1.0, confidence)
    
    def build_equipment_metric_pairs(
        self,
        entities: List[SemanticEntity],
        relations: List[EntityRelation],
    ) -> List[Dict]:
        """Build structured equipment-metric pairs."""
        pairs = []
        
        for rel in relations:
            if rel.relation_type != RelationType.HAS_METRIC:
                continue
            
            source = entities[rel.source_idx]
            target = entities[rel.target_idx]
            
            pair = {
                "equipment": source.normalized,
                "equipment_type": source.attributes.get("equipment_class", "unknown"),
                "metric": target.normalized,
                "metric_value": target.attributes.get("value"),
                "metric_unit": target.attributes.get("unit"),
                "metric_class": target.attributes.get("metric_class"),
                "is_anomaly": target.attributes.get("is_out_of_range", False),
                "confidence": round(rel.confidence, 4),
            }
            pairs.append(pair)
        
        return pairs
