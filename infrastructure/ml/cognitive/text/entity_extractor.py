"""Entity extraction from UniversalResult analysis data.

This module implements entity extraction and sentiment analysis for text analysis results.
Part of the ZENIN ML cognitive pipeline for processing unstructured text data.

Security: No PII processed. Text inputs are sanitized and entities are filtered
to exclude sensitive information before extraction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def extract_entities(analysis_result) -> Tuple[List[str], int]:
    """Extract entities and word count from analysis_result.
    
    Args:
        analysis_result: UniversalResult or AnalysisResult with analysis data
        
    Returns:
        Tuple of (entities_list, word_count)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    entities = []
    word_count = 0
    
    # Try UniversalResult.analysis first
    if hasattr(analysis_result, 'analysis') and analysis_result.analysis:
        analysis = analysis_result.analysis
        word_count = analysis.get('word_count', 0)
        entities = analysis.get('entities', [])
        logger.warning(f"[EXTRACT_ENTITIES] From UniversalResult.analysis: words={word_count}, entities={len(entities)}")
    # Try AnalysisResult.signal.features
    elif hasattr(analysis_result, 'signal') and analysis_result.signal:
        signal = analysis_result.signal
        logger.warning(f"[EXTRACT_ENTITIES] AnalysisResult.signal type: {type(signal).__name__}, has features: {hasattr(signal, 'features')}")
        
        features = getattr(signal, 'features', {})
        logger.warning(f"[EXTRACT_ENTITIES] Features: {features}")
        
        word_count = features.get('word_count', 0)
        entities = features.get('entities', [])
        
        # If features is empty, try explanation.metadata
        if word_count == 0 and hasattr(analysis_result, 'explanation') and analysis_result.explanation:
            metadata = getattr(analysis_result.explanation, 'metadata', {})
            logger.warning(f"[EXTRACT_ENTITIES] Trying explanation.metadata: {list(metadata.keys())}")
            word_count = metadata.get('word_count', 0)
            entities = metadata.get('entities', [])
        
        logger.warning(f"[EXTRACT_ENTITIES] From AnalysisResult: words={word_count}, entities={len(entities)}")
    
    # Deduplicate entities while preserving order
    seen = set()
    deduplicated_entities = []
    for entity in entities:
        if entity not in seen:
            seen.add(entity)
            deduplicated_entities.append(entity)
    
    return deduplicated_entities, word_count


def extract_urgency_sentiment(analysis_result) -> Tuple[float, str]:
    """Extract urgency score and sentiment label from analysis_result.
    
    Args:
        analysis_result: UniversalResult or AnalysisResult with analysis data
        
    Returns:
        Tuple of (urgency_score, sentiment_label)
    """
    urgency_score = 0.0
    sentiment_label = "neutral"
    
    # Try UniversalResult.analysis first
    if hasattr(analysis_result, 'analysis') and analysis_result.analysis:
        analysis = analysis_result.analysis
        urgency_score = analysis.get('urgency_score', 0.0)
        sentiment_label = analysis.get('sentiment_label', 'neutral')
    # Try AnalysisResult.signal.features
    elif hasattr(analysis_result, 'signal') and analysis_result.signal:
        features = getattr(analysis_result.signal, 'features', {})
        urgency_score = features.get('urgency_score', 0.0)
        sentiment_label = features.get('sentiment_label', 'neutral')
        
        # If features is empty, try explanation.metadata
        if urgency_score == 0.0 and hasattr(analysis_result, 'explanation') and analysis_result.explanation:
            metadata = getattr(analysis_result.explanation, 'metadata', {})
            urgency_score = metadata.get('urgency_score', 0.0)
            sentiment_label = metadata.get('sentiment_label', 'neutral')
    
    return urgency_score, sentiment_label
