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
        analysis_result: UniversalResult with analysis data
        
    Returns:
        Tuple of (entities_list, word_count)
    """
    entities = []
    word_count = 0
    
    # Get word count from analysis
    if hasattr(analysis_result, 'analysis') and analysis_result.analysis:
        analysis = analysis_result.analysis
        word_count = analysis.get('word_count', 0)
        entities = analysis.get('entities', [])
    
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
        analysis_result: UniversalResult with analysis data
        
    Returns:
        Tuple of (urgency_score, sentiment_label)
    """
    urgency_score = 0.0
    sentiment_label = "neutral"
    
    if hasattr(analysis_result, 'analysis') and analysis_result.analysis:
        analysis = analysis_result.analysis
        
        # Get urgency score
        urgency_score = analysis.get('urgency_score', 0.0)
        
        # Get sentiment score and derive label
        sentiment_score = analysis.get('sentiment_score', 0.0)
        sentiment_label = 'negative' if sentiment_score < -0.1 else 'positive' if sentiment_score > 0.1 else 'neutral'
    
    return urgency_score, sentiment_label
