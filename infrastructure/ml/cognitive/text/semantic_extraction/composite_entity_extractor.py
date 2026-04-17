"""CompositeEntityExtractor — main adapter implementing EntityExtractorPort.

Orchestrates multiple specialized extractors and relation detection.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from iot_machine_learning.domain.entities.semantic_extraction import (
    EntityType,
    SemanticEnrichmentResult,
    SemanticEntity,
)
from iot_machine_learning.domain.ports.semantic_extraction_port import (
    EntityExtractorPort,
    EntityExtractionResult,
)

from .equipment_extractor import EquipmentExtractor
from .metric_extractor import MetricExtractor
from .relation_detector import RelationDetector


class CompositeEntityExtractor(EntityExtractorPort):
    """Composite extractor coordinating specialized extractors.
    
    Implements EntityExtractorPort for hexagonal architecture.
    Uses multiple extractors and builds entity graph with relations.
    """
    
    def __init__(
        self,
        equipment_extractor: Optional[EquipmentExtractor] = None,
        metric_extractor: Optional[MetricExtractor] = None,
        relation_detector: Optional[RelationDetector] = None,
    ):
        self._equipment = equipment_extractor or EquipmentExtractor()
        self._metric = metric_extractor or MetricExtractor()
        self._relations = relation_detector or RelationDetector()
        self._name = "composite_entity_extractor"
    
    @property
    def extractor_name(self) -> str:
        return self._name
    
    def extract(
        self,
        text: str,
        domain_hint: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> EntityExtractionResult:
        """Extract entities using composite strategy.
        
        Pipeline:
        1. Extract equipment identifiers
        2. Extract metric values with units
        3. Detect relations between entities
        4. Build structured result
        """
        t0 = time.perf_counter()
        
        if not text or not isinstance(text, str):
            return EntityExtractionResult.error("invalid_input")
        
        # Run extractors
        equipment_entities = self._equipment.extract(text, domain_hint)
        metric_entities = self._metric.extract(text, domain_hint)
        
        # Combine all entities
        all_entities: List[SemanticEntity] = []
        all_entities.extend(equipment_entities)
        all_entities.extend(metric_entities)
        
        # Sort by position to maintain document order
        all_entities.sort(key=lambda e: e.start_pos)
        
        # Detect relations
        relations = self._relations.detect_relations(all_entities, text)
        
        # Update entities with relations
        for rel in relations:
            # Add relation to source entity
            entity = all_entities[rel.source_idx]
            updated_relations = list(entity.relations) + [rel.target_idx]
            all_entities[rel.source_idx] = SemanticEntity(
                text=entity.text,
                normalized=entity.normalized,
                entity_type=entity.entity_type,
                start_pos=entity.start_pos,
                end_pos=entity.end_pos,
                confidence=entity.confidence,
                context_window=entity.context_window,
                attributes=entity.attributes,
                relations=updated_relations,
            )
        
        # Compute aggregate confidence
        avg_confidence = (
            sum(e.confidence for e in all_entities) / len(all_entities)
            if all_entities else 0.0
        )
        
        # Determine domain
        detected_domain = self._detect_domain(all_entities, domain_hint)
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        return EntityExtractionResult(
            entities=all_entities,
            relations=relations,
            domain_detected=detected_domain,
            confidence_aggregate=avg_confidence,
            extraction_duration_ms=elapsed_ms,
        )
    
    def supports_domain(self, domain: str) -> bool:
        """Supports any domain with technical text."""
        return True
    
    def is_available(self) -> bool:
        """Always available (regex-based)."""
        return True
    
    def to_enrichment_result(
        self,
        extraction: EntityExtractionResult,
        urgency_context: float = 0.0,
    ) -> SemanticEnrichmentResult:
        """Convert extraction result to enrichment format."""
        # Build equipment-metric pairs
        pairs = self._relations.build_equipment_metric_pairs(
            extraction.entities, extraction.relations
        )
        
        # Identify critical entities
        critical = [e for e in extraction.entities if e.is_critical]
        
        return SemanticEnrichmentResult(
            entities=extraction.entities,
            critical_entities=critical,
            entity_count=len(extraction.entities),
            equipment_metric_pairs=pairs,
            domain_detected=extraction.domain_detected,
            enrichment_confidence=extraction.confidence_aggregate,
        )
    
    def _detect_domain(
        self,
        entities: List[SemanticEntity],
        hint: Optional[str],
    ) -> str:
        """Detect domain from entity composition."""
        if hint:
            return hint
        
        if not entities:
            return "general"
        
        # Count entity types
        type_counts: Dict[EntityType, int] = {}
        for e in entities:
            type_counts[e.entity_type] = type_counts.get(e.entity_type, 0) + 1
        
        # Domain heuristics
        if type_counts.get(EntityType.EQUIPMENT, 0) > 0:
            equip_classes = [
                e.attributes.get("equipment_class", "")
                for e in entities if e.entity_type == EntityType.EQUIPMENT
            ]
            if any(ec in ("compressor", "pump", "valve", "motor")
                   for ec in equip_classes):
                return "industrial"
        
        if type_counts.get(EntityType.METRIC, 0) > 0:
            return "instrumentation"
        
        return "general"
