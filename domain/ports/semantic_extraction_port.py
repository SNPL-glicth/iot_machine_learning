"""EntityExtractorPort — domain port for semantic entity extraction.

Hexagonal Architecture: Domain defines the interface (port),
Infrastructure provides implementations (adapters).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from iot_machine_learning.domain.entities.semantic_extraction import (
    EntityRelation,
    SemanticEntity,
)


@dataclass(frozen=True)
class EntityExtractionResult:
    """Result of entity extraction operation.
    
    Immutable result container with factory methods for common cases.
    """
    entities: List[SemanticEntity] = field(default_factory=list)
    relations: List[EntityRelation] = field(default_factory=list)
    domain_detected: str = "general"
    confidence_aggregate: float = 0.0
    extraction_duration_ms: float = 0.0
    
    def __post_init__(self):
        # Ensure lists are not None
        if self.entities is None:
            object.__setattr__(self, 'entities', [])
        if self.relations is None:
            object.__setattr__(self, 'relations', [])
    
    @property
    def entity_count(self) -> int:
        return len(self.entities)
    
    @property
    def critical_entities(self) -> List[SemanticEntity]:
        """Filter entities marked as critical."""
        return [e for e in self.entities if e.is_critical]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "domain_detected": self.domain_detected,
            "confidence_aggregate": round(self.confidence_aggregate, 4),
            "extraction_duration_ms": round(self.extraction_duration_ms, 4),
            "entity_count": self.entity_count,
        }
    
    @classmethod
    def empty(cls) -> "EntityExtractionResult":
        """Factory method for empty result."""
        return cls()
    
    @classmethod
    def error(cls, message: str) -> "EntityExtractionResult":
        """Factory method for error result."""
        return cls(
            entities=[],
            domain_detected="error",
            confidence_aggregate=0.0,
        )


class EntityExtractorPort(ABC):
    """Port interface for semantic entity extraction.
    
    Implementations (adapters) provide concrete extraction logic:
    - Regex-based extractors
    - NLP/NER-based extractors  
    - Hybrid composite extractors
    
    All implementations must be stateless and thread-safe.
    """
    
    @abstractmethod
    def extract(
        self,
        text: str,
        domain_hint: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> EntityExtractionResult:
        """Extract semantic entities from text.
        
        Args:
            text: Input text to analyze
            domain_hint: Optional domain context ('industrial', 'healthcare', etc.)
            context: Additional extraction context (tenant_id, doc_id, etc.)
            
        Returns:
            EntityExtractionResult with entities and relations
        """
        pass
    
    @abstractmethod
    def supports_domain(self, domain: str) -> bool:
        """Check if extractor supports given domain.
        
        Args:
            domain: Domain string to check
            
        Returns:
            True if this extractor can handle the domain
        """
        pass
    
    @property
    @abstractmethod
    def extractor_name(self) -> str:
        """Unique name for this extractor implementation."""
        pass
    
    def is_available(self) -> bool:
        """Check if extractor is operational (dependencies loaded).
        
        Override in concrete implementations if needed.
        """
        return True


class PriorityScorerPort(ABC):
    """Port for entity prioritization scoring strategies."""
    
    @abstractmethod
    def score(
        self,
        entity: SemanticEntity,
        context: Dict[str, Any],
    ) -> float:
        """Calculate priority score for entity.
        
        Args:
            entity: Entity to score
            context: Scoring context (urgency, domain, etc.)
            
        Returns:
            Priority score (higher = more critical)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Scorer identifier."""
        pass
    
    @property
    @abstractmethod
    def weight(self) -> float:
        """Default weight in multi-scorer aggregation."""
        pass
