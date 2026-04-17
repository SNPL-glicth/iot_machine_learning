"""SemanticEnrichmentPhase — pipeline phase between Perceive and Analyze.

Enriches metadata with structured semantic entities for TEXT inputs.
Integrates seamlessly with UniversalAnalysisEngine pipeline.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from iot_machine_learning.domain.entities.semantic_extraction import (
    EnrichmentContext,
    SemanticEnrichmentResult,
)
from iot_machine_learning.domain.ports.semantic_extraction_port import EntityExtractorPort

from iot_machine_learning.infrastructure.ml.cognitive.text.semantic_extraction import (
    ExtractorFactory,
)

from .types import InputType


class SemanticEnrichmentPhase:
    """Phase 1.5: Semantic enrichment for text inputs.
    
    Positioned between PerceivePhase and AnalyzePhase.
    Extracts structured entities and adds them to metadata
    for downstream perception generation.
    """
    
    def __init__(
        self,
        extractor: Optional[EntityExtractorPort] = None,
        enabled: bool = True,
        min_text_length: int = 20,
    ):
        """Initialize enrichment phase.
        
        Args:
            extractor: Entity extractor implementation (default: composite)
            enabled: Feature flag to enable/disable enrichment
            min_text_length: Minimum text length to trigger extraction
        """
        self._extractor = extractor or ExtractorFactory.create_composite_extractor()
        self._enabled = enabled
        self._min_length = min_text_length
    
    def execute(
        self,
        raw_data: Any,
        input_type: InputType,
        metadata: Dict[str, Any],
        domain: str,
        timing: Dict[str, float],
    ) -> Tuple[Dict[str, Any], Optional[SemanticEnrichmentResult]]:
        """Execute semantic enrichment.
        
        Args:
            raw_data: Original input (text for TEXT inputs)
            input_type: Detected input type
            metadata: Current metadata (will be enriched)
            domain: Detected domain
            timing: Pipeline timing dict
            
        Returns:
            Tuple of (enriched_metadata, enrichment_result)
        """
        t0 = time.monotonic()
        
        # Skip if not enabled or not text
        if not self._enabled or input_type != InputType.TEXT:
            timing["enrich"] = 0.0
            # OBSERVABILITY: Track skip reason
            from .....ml_service.metrics.observability import get_observability
            reason = "disabled" if not self._enabled else "not_text"
            get_observability().semantic.record_skip(reason)
            return metadata, None
        
        # Skip if text too short
        text = str(raw_data) if isinstance(raw_data, str) else ""
        if len(text) < self._min_length:
            timing["enrich"] = 0.0
            # OBSERVABILITY: Track skip due to short text
            from .....ml_service.metrics.observability import get_observability
            get_observability().semantic.record_skip(f"short_text_{len(text)}")
            return metadata, None
        
        try:
            # Build enrichment context
            ctx = EnrichmentContext(
                domain=domain,
                urgency_score=metadata.get("urgency_score", 0.0),
                document_position=self._detect_position(metadata),
            )
            
            # Run extraction
            extraction = self._extractor.extract(text, domain_hint=domain)
            
            # Convert to enrichment result
            result = self._extractor.to_enrichment_result(
                extraction, urgency_context=ctx.urgency_score
            )
            
            # Prioritize entities
            from iot_machine_learning.application.semantic_extraction import (
                EntityPrioritizer,
            )
            prioritizer = EntityPrioritizer()
            prioritized = prioritizer.prioritize(result.entities, ctx)
            
            # Update result with critical entities
            result = prioritized.to_enrichment_result(result)
            
            # Enrich metadata
            enriched = dict(metadata)
            enriched["semantic_enrichment"] = result.to_dict()
            enriched["has_semantic_entities"] = True
            enriched["entity_count"] = result.entity_count
            
            # Add summary for quick access
            if result.equipment_metric_pairs:
                enriched["equipment_metric_pairs"] = result.equipment_metric_pairs
            
            timing["enrich"] = (time.monotonic() - t0) * 1000
            
            # OBSERVABILITY: Track semantic enrichment execution
            from .....ml_service.metrics.observability import get_observability
            has_critical = any(e.is_critical for e in result.entities) if result.entities else False
            get_observability().semantic.record_execution(result.entity_count, has_critical)
            
            return enriched, result
            
        except Exception as e:
            # Graceful degradation — don't break pipeline
            timing["enrich"] = (time.monotonic() - t0) * 1000
            enriched = dict(metadata)
            enriched["semantic_enrichment_error"] = str(e)
            # OBSERVABILITY: Track semantic enrichment error
            from .....ml_service.metrics.observability import get_observability
            get_observability().semantic.record_error(str(e))
            return enriched, None
    
    def _detect_position(self, metadata: Dict[str, Any]) -> str:
        """Detect document position from metadata."""
        # Could be extended with actual position detection
        return "body"
    
    @property
    def is_enabled(self) -> bool:
        """Check if enrichment is enabled."""
        return self._enabled
    
    def enable(self) -> None:
        """Enable enrichment."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable enrichment."""
        self._enabled = False
