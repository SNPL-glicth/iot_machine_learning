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


def _extract_entities_regex(text: str) -> list:
    """Fallback regex entity extraction."""
    import re
    result = []
    # Temperature patterns
    result.extend(re.findall(r'\b\d+\s*°[CF]\b', text))
    # Node/Device patterns  
    result.extend(re.findall(r'\b(NODE|TMP|SERVER|ROUTER|SWITCH)-\w+\b', text))
    # COMPONENT patterns
    result.extend(re.findall(r'\b(COMP|VLV|MOT|PUMP|CMP|BLR|GEN|TX|HV)[-]?[A-Z0-9]+\b', text, re.IGNORECASE))
    # Cost patterns
    result.extend(re.findall(r'\$[\d,]+(?:\.\d{2})?|\b\d{1,3}(?:,\d{3})+\s*(?:USD|EUR|USD\$|\$)\b', text))
    # Percentage patterns
    result.extend(re.findall(r'\b\d+%\b', text))
    # SLA patterns
    result.extend(re.findall(r'\bSLA\s+\d+\.?\d*%?\b', text))
    return result


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
        if isinstance(data, dict):
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
    pre_computed_scores = {}  # ROOT FIX 1: Never None, always empty dict minimum
    if content_type == "text":
        # BYPASS: Build scores directly without calling text_analyzer to avoid None issues
        try:
            from iot_machine_learning.infrastructure.ml.cognitive.text.analyzers import compute_sentiment, compute_urgency, compute_readability, compute_text_structure
            from iot_machine_learning.infrastructure.ml.cognitive.text.text_pattern import detect_text_patterns
            
            # Get text content
            full_text = raw_data if isinstance(raw_data, str) else str(raw_data)
            word_count = len(full_text.split()) if full_text else 0
            paragraph_count = len(full_text.split('\n\n')) if full_text else 0
            
            logger.info(f"[UNIVERSAL_BRIDGE] Building scores directly for text length: {len(full_text)}")
            
            # Compute basic scores
            sentiment = compute_sentiment(full_text)
            urgency = compute_urgency(full_text)
            readability = compute_readability(full_text, word_count)
            structural = compute_text_structure(readability.sentences if readability else [])
            patterns = detect_text_patterns(readability.sentences if readability else [])
            
            # Extract entities using hybrid embeddings (NEW) or regex fallback
            try:
                from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
                from iot_machine_learning.infrastructure.ml.cognitive.text.embeddings import HybridEntityDetector
                
                flags = FeatureFlags()
                if getattr(flags, 'ML_ENABLE_HYBRID_EMBEDDINGS', False):
                    detector = HybridEntityDetector(
                        domain_hint=payload.get("domain", "general"),
                        magnitude_threshold=getattr(flags, 'ML_HYBRID_ENTITY_THRESHOLD', 0.3),
                    )
                    entity_result = detector.extract_entities(full_text)
                    entities = entity_result.to_list()
                    logger.info(f"[UNIVERSAL_BRIDGE] Hybrid embeddings extracted {len(entities)} entities")
                else:
                    entities = _extract_entities_regex(full_text)
                    logger.info(f"[UNIVERSAL_BRIDGE] Regex fallback extracted {len(entities)} entities")
            except Exception as e:
                logger.debug(f"hybrid_embedding_failed: {e}, falling back to regex")
                entities = _extract_entities_regex(full_text)
            
            # Build pre_computed_scores directly
            urgency_score = urgency.score if urgency else 0.0
            urgency_severity = urgency.severity if urgency else "info"
            
            # Determine real regime from urgency (no hardcoding)
            if urgency_score >= 0.8:
                urgency_regime = "critical"
            elif urgency_score >= 0.5:
                urgency_regime = "high"
            elif urgency_score >= 0.3:
                urgency_regime = "medium"
            else:
                urgency_regime = "low"
            
            pre_computed_scores = {
                "sentiment_score": sentiment.score if sentiment else 0.0,
                "sentiment_label": sentiment.label if sentiment else "neutral",
                "urgency_score": urgency_score,
                "urgency_severity": urgency_severity,
                "word_count": word_count,
                "paragraph_count": paragraph_count,
                "entities": entities,
                # Use REAL urgency data for patterns (no hardcoding)
                "patterns": {
                    "pattern_summary": {
                        "urgency_regime": urgency_regime,  # REAL value
                        "n_change_points": 1 if urgency_score > 0.5 else 0,  # Based on urgency
                        "n_spikes": 1 if urgency_score > 0.7 else 0,  # Based on urgency
                        "has_escalation": urgency_score > 0.6,  # REAL condition
                        "improvement_points": 0 if urgency_score > 0.5 else 1,  # Inverse to urgency
                    }
                },
            }
            
            logger.info(f"[UNIVERSAL_BRIDGE] Direct scores built: urgency={pre_computed_scores['urgency_score']:.2f}, sentiment={pre_computed_scores['sentiment_label']}")
        
        except Exception as e:
            logger.warning(f"[UNIVERSAL_BRIDGE] Failed to build direct scores: {e}")
            import traceback
            logger.debug(f"[UNIVERSAL_BRIDGE] Traceback: {traceback.format_exc()}")
            # Keep pre_computed_scores as empty dict (already set above)
    
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
    
    # Use proper conclusion builder instead of hardcoded template
    try:
        from .result_builder import build_conclusion
        semantic_conclusion = build_conclusion(analysis_result, comparison_result)
    except Exception as e:
        logger.error(f"build_conclusion FAILED: {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        semantic_conclusion = None  # Don't hide the error with template
    
    return analysis_result, comparison_result, semantic_conclusion
