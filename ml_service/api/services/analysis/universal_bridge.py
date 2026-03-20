"""Bridge between DocumentAnalyzer and UniversalAnalysisEngine.

Handles universal input detection, engine invocation, and comparative analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from iot_machine_learning.infrastructure.ml.cognitive.universal import (
    UniversalAnalysisEngine,
    UniversalComparativeEngine,
    UniversalInput,
    UniversalContext,
    ComparisonContext,
    InputType,
)
from iot_machine_learning.infrastructure.ml.cognitive.universal.analysis.input_detector import (
    detect_input_type,
)

logger = logging.getLogger(__name__)


def extract_raw_data(payload: Dict[str, Any], content_type: str) -> Any:
    """Extract raw data from normalized payload.
    
    Args:
        payload: Normalized document payload
        content_type: Content type hint
        
    Returns:
        Raw data for analysis
    """
    if "full_text" in payload:
        return payload["full_text"]
    elif "text" in payload:
        return payload["text"]
    elif "content" in payload:
        return payload["content"]
    elif "values" in payload:
        return payload["values"]
    elif "data" in payload:
        return payload["data"]
    else:
        return payload


def analyze_with_universal(
    document_id: str,
    content_type: str,
    payload: Dict[str, Any],
    tenant_id: str,
    analysis_engine: UniversalAnalysisEngine,
    comparative_engine: Optional[UniversalComparativeEngine],
    cognitive_memory: Optional[object],
) -> tuple[Any, Optional[Any]]:
    """Analyze using UniversalAnalysisEngine + UniversalComparativeEngine.
    
    Args:
        document_id: Document identifier
        content_type: Content type hint
        payload: Normalized payload
        tenant_id: Tenant identifier
        analysis_engine: Universal analysis engine
        comparative_engine: Optional comparative engine
        cognitive_memory: Optional cognitive memory port
        
    Returns:
        Tuple of (analysis_result, comparison_result)
    """
    # Extract raw data from payload
    raw_data = extract_raw_data(payload, content_type)
    
    # Auto-detect input type if not explicitly provided
    detected_type = detect_input_type(raw_data)
    
    # Build universal input
    universal_input = UniversalInput(
        raw_data=raw_data,
        detected_type=detected_type,
        metadata=payload.get("metadata", {}),
        domain_hint=payload.get("domain", ""),
        series_id=document_id,
    )
    
    # Build context
    context = UniversalContext(
        series_id=document_id,
        tenant_id=tenant_id,
        cognitive_memory=cognitive_memory,
        domain_hint=payload.get("domain", ""),
        budget_ms=2000.0,
    )
    
    # Run analysis
    analysis_result = analysis_engine.analyze(universal_input, context)
    
    # Run comparative analysis if memory available
    comparison_result = None
    if comparative_engine and cognitive_memory:
        try:
            comp_ctx = ComparisonContext(
                current_result=analysis_result,
                series_id=document_id,
                tenant_id=tenant_id,
                cognitive_memory=cognitive_memory,
                domain=analysis_result.domain,
            )
            comparison_result = comparative_engine.compare(comp_ctx)
        except Exception as e:
            logger.warning(f"comparative_analysis_failed: {e}")
    
    return analysis_result, comparison_result
