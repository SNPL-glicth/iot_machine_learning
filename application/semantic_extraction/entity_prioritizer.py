"""EntityPrioritizer — multi-factor entity ranking engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from iot_machine_learning.domain.entities.semantic_extraction import (
    EntityType,
    SemanticEntity,
    SemanticEnrichmentResult,
    EnrichmentContext,
)


class PriorityScorer(Protocol):
    """Protocol for entity scoring strategies."""
    name: str
    weight: float
    
    def score(self, entity: SemanticEntity, context: Dict[str, Any]) -> float:
        ...


@dataclass(frozen=True)
class RankedEntity:
    """Entity with computed priority score."""
    entity: SemanticEntity
    score: float
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity": self.entity.to_dict(),
            "score": round(self.score, 4),
            "score_breakdown": self.score_breakdown,
        }


@dataclass(frozen=True)
class PrioritizationResult:
    """Result of entity prioritization."""
    ranked_entities: List[RankedEntity]
    critical_threshold: float
    
    @property
    def top_entities(self) -> List[RankedEntity]:
        """Get top 5 entities by score."""
        return self.ranked_entities[:5]
    
    @property
    def critical_entities(self) -> List[RankedEntity]:
        """Get entities above critical threshold."""
        return [re for re in self.ranked_entities if re.score >= self.critical_threshold]
    
    def to_enrichment_result(
        self,
        original: SemanticEnrichmentResult,
    ) -> SemanticEnrichmentResult:
        """Convert back to enrichment result with rankings."""
        critical = [re.entity for re in self.critical_entities]
        return SemanticEnrichmentResult(
            entities=original.entities,
            critical_entities=critical,
            entity_count=original.entity_count,
            equipment_metric_pairs=original.equipment_metric_pairs,
            domain_detected=original.domain_detected,
            enrichment_confidence=original.enrichment_confidence,
        )


class EntityPrioritizer:
    """Multi-factor entity prioritization engine.
    
    Uses Strategy pattern for composable scoring.
    """
    
    # Type-based base scores
    TYPE_WEIGHTS = {
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
    
    # Urgency keywords for context scoring
    URGENCY_KEYWORDS = [
        "critical", "failure", "alarm", "emergency", "shutdown",
        "trip", "overheat", "overpressure", "leak", "rupture",
        "anomaly", "deviation", "fault", "error", "warning",
    ]
    
    def __init__(self, critical_threshold: float = 100.0):
        self.critical_threshold = critical_threshold
    
    def prioritize(
        self,
        entities: List[SemanticEntity],
        context: EnrichmentContext,
    ) -> PrioritizationResult:
        """Rank entities by criticality.
        
        Algorithm:
        1. Base score by entity type
        2. Boost for out-of-range metrics
        3. Boost for proximity to urgency keywords
        4. Boost for document position
        5. Normalize by confidence
        """
        ranked = []
        ctx_dict = context.to_dict()
        
        for entity in entities:
            breakdown = {}
            
            # 1. Type-based base score
            type_score = self._type_score(entity)
            breakdown["type"] = type_score
            
            # 2. Metric anomaly boost
            anomaly_score = self._anomaly_score(entity)
            breakdown["anomaly"] = anomaly_score
            
            # 3. Context proximity to urgency
            urgency_score = self._urgency_proximity_score(entity)
            breakdown["urgency_context"] = urgency_score
            
            # 4. Document position boost
            position_score = self._position_score(entity, context.document_position)
            breakdown["document_position"] = position_score
            
            # 5. Confidence normalization
            confidence_factor = 0.5 + (0.5 * entity.confidence)
            
            # Total score
            total = (type_score + anomaly_score + urgency_score + position_score)
            total *= confidence_factor
            
            ranked.append(RankedEntity(
                entity=entity,
                score=total,
                score_breakdown=breakdown,
            ))
        
        # Sort by score descending
        ranked.sort(key=lambda x: x.score, reverse=True)
        
        return PrioritizationResult(
            ranked_entities=ranked,
            critical_threshold=self.critical_threshold,
        )
    
    def _type_score(self, entity: SemanticEntity) -> float:
        """Base score from entity type."""
        return float(self.TYPE_WEIGHTS.get(entity.entity_type, 50))
    
    def _anomaly_score(self, entity: SemanticEntity) -> float:
        """Score boost for anomalous metrics."""
        if entity.entity_type != EntityType.METRIC:
            return 0.0
        
        attrs = entity.attributes
        if not attrs.get("is_out_of_range", False):
            return 0.0
        
        # Scale by deviation severity
        deviation = abs(attrs.get("deviation_percent", 0))
        return min(80, 40 + (deviation * 0.4))  # Max 80 bonus
    
    def _urgency_proximity_score(self, entity: SemanticEntity) -> float:
        """Score boost for entities near urgency keywords."""
        if not entity.context_window:
            return 0.0
        
        window_lower = entity.context_window.lower()
        matches = sum(1 for kw in self.URGENCY_KEYWORDS if kw in window_lower)
        
        return min(50, matches * 10)  # 10 per match, max 50
    
    def _position_score(self, entity: SemanticEntity, position: str) -> float:
        """Score boost based on document position."""
        position_multipliers = {
            "title": 40,
            "opening": 30,
            "conclusion": 20,
            "body": 0,
        }
        return float(position_multipliers.get(position, 0))
