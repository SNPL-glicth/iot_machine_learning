"""Universal perception collector — dispatcher to type-specific analyzers.

CRITICAL: Reuses existing infrastructure/ml/ components for numeric analysis.
Does NOT reinvent analyzers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ...analysis.types import EnginePerception
from .types import InputType
# text_perception_collector.py deleted - text perception removed
# from .text_perception_collector import collect_text_perceptions
from .numeric_perception_collector import collect_numeric_perceptions, collect_tabular_perceptions

logger = logging.getLogger(__name__)


class UniversalPerceptionCollector:
    """Maps any input type to List[EnginePerception].

    Dispatches to appropriate sub-analyzers based on InputType.
    
    TEXT: Reuses text_sentiment, text_urgency, text_readability, text_structural, text_pattern
    NUMERIC: Wraps existing infrastructure/ml/ components as EnginePerceptions
    TABULAR: Analyzes first numeric column
    MIXED: Hybrid approach
    """

    def collect(
        self,
        raw_data: Any,
        input_type: InputType,
        metadata: Dict[str, Any],
        pre_computed_scores: Optional[Dict[str, Any]] = None,
    ) -> List[EnginePerception]:
        """Build EnginePerception list from input.

        Args:
            raw_data: Original input
            input_type: Detected InputType
            metadata: From input_detector
            pre_computed_scores: Scores from ml_service analyzers (if available)

        Returns:
            List of EnginePerception (one per sub-analyzer)
        """
        if input_type == InputType.TEXT:
            # Merge pre_computed_scores with metadata (includes semantic_enrichment)
            merged_scores = {**(pre_computed_scores or {})}
            # Add semantic_enrichment from metadata if present
            if metadata and "semantic_enrichment" in metadata:
                merged_scores["semantic_enrichment"] = metadata["semantic_enrichment"]
            return self._collect_text(merged_scores)
        
        if input_type == InputType.NUMERIC:
            return self._collect_numeric(raw_data, None, metadata)
        
        if input_type == InputType.TABULAR:
            return self._collect_tabular(raw_data, metadata)
        
        if input_type == InputType.MIXED:
            return self._collect_mixed(raw_data, metadata)
        
        return []

    def _collect_text(
        self,
        scores: Dict[str, Any],
    ) -> List[EnginePerception]:
        """Delegate to text perception logic."""
        # text_perception_collector.py deleted - return empty list
        return []

    def _collect_numeric(
        self,
        values: list,
        timestamps: Optional[list],
        metadata: Dict[str, Any],
    ) -> List[EnginePerception]:
        """Build perceptions for numeric series."""
        return collect_numeric_perceptions(values, timestamps, metadata)

    def _collect_tabular(
        self,
        data: dict,
        metadata: Dict[str, Any],
    ) -> List[EnginePerception]:
        """Build perceptions for tabular data."""
        return collect_tabular_perceptions(data, metadata)

    def _collect_mixed(
        self,
        raw_data: Any,
        metadata: Dict[str, Any],
    ) -> List[EnginePerception]:
        """Hybrid collection for mixed-type data."""
        return []
