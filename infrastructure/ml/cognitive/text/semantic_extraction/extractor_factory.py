"""ExtractorFactory — dependency injection factory for extractors."""

from __future__ import annotations

from typing import Optional

from iot_machine_learning.domain.ports.semantic_extraction_port import EntityExtractorPort

from .composite_entity_extractor import CompositeEntityExtractor
from .equipment_extractor import EquipmentExtractor
from .metric_extractor import MetricExtractor
from .relation_detector import RelationDetector


class ExtractorFactory:
    """Factory for creating configured entity extractors.
    
    Centralizes extractor configuration and enables:
    - Dependency injection
    - Test mocking
    - Feature flag integration
    """
    
    @staticmethod
    def create_composite_extractor(
        enable_equipment: bool = True,
        enable_metrics: bool = True,
        enable_relations: bool = True,
    ) -> EntityExtractorPort:
        """Create fully configured composite extractor.
        
        Args:
            enable_equipment: Include equipment extraction
            enable_metrics: Include metric extraction
            enable_relations: Include relation detection
            
        Returns:
            Configured CompositeEntityExtractor
        """
        equipment = EquipmentExtractor() if enable_equipment else None
        metric = MetricExtractor() if enable_metrics else None
        relations = RelationDetector() if enable_relations else None
        
        return CompositeEntityExtractor(
            equipment_extractor=equipment,
            metric_extractor=metric,
            relation_detector=relations,
        )
    
    @staticmethod
    def create_equipment_only() -> EntityExtractorPort:
        """Create extractor for equipment only (lightweight)."""
        return CompositeEntityExtractor(
            equipment_extractor=EquipmentExtractor(),
            metric_extractor=None,
            relation_detector=None,
        )
    
    @staticmethod
    def create_metrics_only() -> EntityExtractorPort:
        """Create extractor for metrics only (lightweight)."""
        return CompositeEntityExtractor(
            equipment_extractor=None,
            metric_extractor=MetricExtractor(),
            relation_detector=None,
        )
    
    @staticmethod
    def create_with_mock(
        mock_extractor: Optional[EntityExtractorPort] = None,
    ) -> EntityExtractorPort:
        """Create extractor with mock for testing."""
        if mock_extractor:
            return mock_extractor
        return ExtractorFactory.create_composite_extractor()
