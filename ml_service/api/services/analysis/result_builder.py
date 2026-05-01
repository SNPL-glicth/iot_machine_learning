"""Result builder orchestrator for document analysis.

Thin orchestrator that delegates to specialized modules.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from iot_machine_learning.domain.services.conclusion_formatter import format_conclusion, format_simple_conclusion
from iot_machine_learning.infrastructure.ml.cognitive.text.entity_extractor import (
    extract_entities,
    extract_urgency_sentiment,
)
from .output_assembler import build_output_dict as assemble_output_dict


def build_conclusion(
    analysis_result,
    comparison_result: Optional[object] = None,
) -> str:
    """Build human-readable conclusion from analysis + comparison.
    
    Args:
        analysis_result: UniversalAnalysisResult from engine
        comparison_result: Optional comparison result
        
    Returns:
        Multi-line conclusion string
    """
    # Check if result has explanation (UniversalResult)
    if not hasattr(analysis_result, 'explanation'):
        return format_simple_conclusion(analysis_result)
    
    entity_list, word_count = extract_entities(analysis_result)
    urgency_score, sentiment_label = extract_urgency_sentiment(analysis_result)
    entities_dict = {
        "entities": entity_list,
        "word_count": word_count,
        "urgency_score": urgency_score,
        "sentiment_label": sentiment_label,
    }
    return format_conclusion(analysis_result, entities_dict, comparison_result)


def build_output_dict(
    analysis_result,
    comparison_result: Optional[object],
    raw_data: Any,
    semantic_conclusion: Optional[str] = None,
) -> Dict[str, Any]:
    """Build structured output dictionary from analysis results.
    
    Args:
        analysis_result: UniversalAnalysisResult from engine
        comparison_result: Optional comparison result
        raw_data: Original raw data
        semantic_conclusion: Optional semantic conclusion (legacy)
        
    Returns:
        Structured output dictionary
    """
    return assemble_output_dict(
        analysis_result, comparison_result, raw_data, semantic_conclusion
    )
