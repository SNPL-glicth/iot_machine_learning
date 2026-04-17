"""Entity relations for semantic graph construction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class RelationType(Enum):
    """Types of semantic relations between entities."""
    HAS_METRIC = "has_metric"           # Equipment → Metric
    LOCATED_AT = "located_at"         # Equipment → Location
    PART_OF = "part_of"               # Component → Equipment
    MEASURED_AT = "measured_at"        # Metric → Location
    OCCURRED_DURING = "occurred_during"  # Event → Temporal
    CAUSED_BY = "caused_by"           # Effect → Cause
    TRIGGERS = "triggers"               # Alert → Action
    ASSOCIATED_WITH = "associated_with"  # General association
    OPERATED_BY = "operated_by"        # Equipment → Personnel
    MONITORED_BY = "monitored_by"      # Equipment/System → System


@dataclass(frozen=True)
class EntityRelation:
    """Relation between two semantic entities.
    
    Relations are stored by entity indices in the entity list,
    forming a semantic graph.
    """
    source_idx: int
    target_idx: int
    relation_type: RelationType
    confidence: float = 0.8
    context: str = ""  # Text evidence for relation
    
    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            object.__setattr__(
                self, 'confidence', max(0.0, min(1.0, self.confidence))
            )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_idx": self.source_idx,
            "target_idx": self.target_idx,
            "relation_type": self.relation_type.value,
            "confidence": round(self.confidence, 4),
            "context": self.context,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityRelation":
        return cls(
            source_idx=int(data.get("source_idx", 0)),
            target_idx=int(data.get("target_idx", 0)),
            relation_type=RelationType(data.get("relation_type", "associated_with")),
            confidence=float(data.get("confidence", 0.8)),
            context=str(data.get("context", "")),
        )
    
    def is_equipment_metric(self) -> bool:
        """Check if this is an equipment-metric relation."""
        return self.relation_type == RelationType.HAS_METRIC
    
    def is_location_based(self) -> bool:
        """Check if this is a location-based relation."""
        return self.relation_type in (
            RelationType.LOCATED_AT,
            RelationType.MEASURED_AT,
        )
