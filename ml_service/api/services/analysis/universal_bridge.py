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
    # DEBUG: Log payload structure
    logger.info(f"[UNIVERSAL_BRIDGE] extract_raw_data: content_type={content_type}, payload_keys={list(payload.keys())}")
    
    if "data" in payload:
        data = payload["data"]
        logger.info(f"[UNIVERSAL_BRIDGE] Found 'data' key with: {list(data.keys())}")
        if "full_text" in data:
            full_text = data["full_text"]
            logger.info(f"[UNIVERSAL_BRIDGE] Found full_text: length={len(full_text)}, preview={full_text[:100]!r}")
            # CRITICAL FIX: Return the actual text string, not the dict
            return full_text
    
    if "full_text" in payload:
        logger.info(f"[UNIVERSAL_BRIDGE] Found top-level full_text")
        return payload["full_text"]
    elif "text" in payload:
        logger.info(f"[UNIVERSAL_BRIDGE] Found top-level text")
        return payload["text"]
    elif "content" in payload:
        logger.info(f"[UNIVERSAL_BRIDGE] Found top-level content")
        return payload["content"]
    elif "values" in payload:
        logger.info(f"[UNIVERSAL_BRIDGE] Found top-level values")
        return payload["values"]
    elif "data" in payload:
        logger.warning(f"[UNIVERSAL_BRIDGE] Returning data dict as-is (should not happen for text)")
        return payload["data"]
    else:
        logger.warning(f"[UNIVERSAL_BRIDGE] No recognized data keys found, returning payload as-is: {list(payload.keys())}")
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
    
    # Extract pre-computed scores for text content
    pre_computed_scores = None
    if content_type == "text" and "data" in payload:
        data = payload["data"]
        # For text content, we need to run the ml_service text analyzers first
        # to get the scores that UniversalPerceptionCollector expects
        try:
            from ..analyzers.text_analyzer import analyze_text_document
            text_result = analyze_text_document(document_id, payload)
            
            # Extract the scores from the text analysis result
            analysis = text_result.get("analysis", {})
            
            # Handle different analysis result formats
            sentiment = analysis.get("sentiment", {})
            if isinstance(sentiment, str):
                sentiment = {"score": 0.0, "label": sentiment}
            elif not isinstance(sentiment, dict):
                sentiment = {"score": 0.0, "label": "neutral"}
            
            urgency = analysis.get("urgency", {})
            if isinstance(urgency, str):
                urgency = {"score": 0.0, "level": urgency}
            elif not isinstance(urgency, dict):
                urgency = {"score": 0.0, "level": "info"}
            
            # Extract urgency_score and severity from the correct locations
            urgency_score = analysis.get("urgency_score", urgency.get("score", 0.0))
            urgency_severity = analysis.get("urgency_severity", urgency.get("level", "info"))
            
            pre_computed_scores = {
                "sentiment_score": sentiment.get("score", 0.0),
                "sentiment_label": sentiment.get("label", "neutral"),
                "urgency_score": urgency_score,
                "urgency_severity": urgency_severity,
                "word_count": data.get("word_count", 0),
                "paragraph_count": data.get("paragraph_count", 0),
            }
            
            logger.info(f"[UNIVERSAL_BRIDGE] Extracted pre-computed scores: {list(pre_computed_scores.keys())}")
            logger.debug(f"[UNIVERSAL_BRIDGE] Sentiment: {sentiment}, Urgency: {urgency}")
            logger.info(f"[UNIVERSAL_BRIDGE] Pre-computed urgency_score: {pre_computed_scores.get('urgency_score', 'N/A')}")
            logger.info(f"[UNIVERSAL_BRIDGE] Pre-computed sentiment_label: {pre_computed_scores.get('sentiment_label', 'N/A')}")
        except Exception as e:
            logger.warning(f"[UNIVERSAL_BRIDGE] Failed to extract pre-computed scores: {e}")
            import traceback
            logger.debug(f"[UNIVERSAL_BRIDGE] Traceback: {traceback.format_exc()}")
            pre_computed_scores = None
    
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
    
    # Run analysis with pre-computed scores
    logger.info(f"[STAGE-6] UniversalAnalysisEngine.analyze called")
    analysis_result = analysis_engine.analyze(
        raw_data=universal_input.raw_data,
        ctx=context,
        pre_computed_scores=pre_computed_scores,
    )
    
    logger.info(f"[STAGE-7] result domain={getattr(analysis_result, 'domain', 'N/A')}, confidence={getattr(analysis_result, 'confidence', 'N/A')}")
    
    # CRITICAL FIX: Build semantic conclusion with entity extraction for text content
    semantic_conclusion = None
    if content_type == "text" and "data" in payload:
        try:
            from ..analyzers.text_analyzer import analyze_text_document
            text_result = analyze_text_document(document_id, payload)
            
            # Extract the semantic conclusion from text analyzer
            semantic_conclusion = text_result.get("conclusion", "")
            if semantic_conclusion and len(semantic_conclusion.strip()) > 50:
                logger.info(f"[UNIVERSAL_BRIDGE] Using semantic conclusion with entity extraction")
                logger.debug(f"[UNIVERSAL_BRIDGE] Semantic conclusion preview: {semantic_conclusion[:200]}...")
            else:
                semantic_conclusion = None
        except Exception as e:
            logger.warning(f"[UNIVERSAL_BRIDGE] Failed to build semantic conclusion: {e}")
            semantic_conclusion = None
    
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
    
    return analysis_result, comparison_result, semantic_conclusion
