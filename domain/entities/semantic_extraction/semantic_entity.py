"""SemanticEntity — domain entity for structured semantic extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EntityType(Enum):
    """Semantic entity types for industrial/technical text."""
    EQUIPMENT = "equipment"        # C-12, V-23, PUMP-A, COMP-01
    COMPONENT = "component"        # valve, motor, compressor, bearing
    METRIC = "metric"              # 3401 PSI, 441.2°C, 85%
    LOCATION = "location"          # Sector 7, Building A, Unit 3
    TEMPORAL = "temporal"          # 2024-01-15, Q3 2024, yesterday
    OPERATIONAL = "operational"    # shutdown, startup, maintenance, trip
    ALERT = "alert"                # critical, warning, error, alarm
    PERSONNEL = "personnel"        # Operator, Engineer, Supervisor
    SYSTEM = "system"              # HVAC, DCS, SCADA, BMS


@dataclass(frozen=True)
class SemanticEntity:
    """Structured semantic entity extracted from text.
    
    Attributes:
        text: Original text span (e.g., "C-12")
        normalized: Normalized form (e.g., "C12")
        entity_type: Semantic category
        start_pos: Character start position
        end_pos: Character end position
        confidence: Extraction confidence [0.0, 1.0]
        context_window: Surrounding text (~50 chars)
        attributes: Type-specific attributes (MetricAttributes, etc.)
        relations: Indices of related entities
    """
    text: str
    normalized: str
    entity_type: EntityType
    start_pos: int
    end_pos: int
    confidence: float = 0.8
    context_window: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    relations: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        # Validate confidence range
        if not 0.0 <= self.confidence <= 1.0:
            object.__setattr__(
                self, 'confidence', max(0.0, min(1.0, self.confidence))
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "text": self.text,
            "normalized": self.normalized,
            "entity_type": self.entity_type.value,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": round(self.confidence, 4),
            "context_window": self.context_window,
            "attributes": self.attributes,
            "relations": self.relations,
        }
    
    @property
    def is_critical(self) -> bool:
        """Check if entity is potentially critical."""
        if self.entity_type in (EntityType.ALERT, EntityType.OPERATIONAL):
            return True
        if self.entity_type == EntityType.METRIC:
            return self.attributes.get("is_out_of_range", False)
        return False
    
    @property
    def is_equipment_metric_pair(self) -> bool:
        """Check if this entity pairs equipment with metric."""
        return (
            self.entity_type == EntityType.METRIC 
            and len(self.relations) > 0
        )


@dataclass(frozen=True)
class EnrichmentContext:
    """Context for semantic enrichment decisions."""
    domain: str = "general"
    urgency_score: float = 0.0
    tenant_id: str = ""
    document_position: str = "body"  # title, opening, body, conclusion
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "urgency_score": self.urgency_score,
            "document_position": self.document_position,
        }


@dataclass(frozen=True)
class SemanticEnrichmentResult:
    """Result of semantic enrichment phase."""
    entities: List[SemanticEntity]
    critical_entities: List[SemanticEntity]
    entity_count: int
    equipment_metric_pairs: List[Dict[str, Any]]
    domain_detected: str
    enrichment_confidence: float
    
    def __post_init__(self):
        # Ensure critical_entities is subset of entities
        if not all(e in self.entities for e in self.critical_entities):
            object.__setattr__(
                self, 'critical_entities', 
                [e for e in self.critical_entities if e in self.entities]
            )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "critical_entities": [e.to_dict() for e in self.critical_entities],
            "entity_count": self.entity_count,
            "equipment_metric_pairs": self.equipment_metric_pairs,
            "domain_detected": self.domain_detected,
            "enrichment_confidence": round(self.enrichment_confidence, 4),
        }
    
    def to_perception_metadata(self) -> Dict[str, Any]:
        """Convert to metadata format for EnginePerception."""
        return {
            "semantic_enrichment": self.to_dict(),
            "entity_types_found": list(set(
                e.entity_type.value for e in self.entities
            )),
            "has_critical_equipment": any(
                e.entity_type == EntityType.EQUIPMENT 
                for e in self.critical_entities
            ),
            "equipment_metric_count": len(self.equipment_metric_pairs),
        }
