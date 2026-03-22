"""Conclusion formatting from UniversalResult data."""

from __future__ import annotations

from typing import Any, Dict, Optional

from iot_machine_learning.infrastructure.ml.cognitive.text.entity_extractor import extract_entities, extract_urgency_sentiment
from .action_recommender import recommend_actions


def format_conclusion(analysis_result, comparison_result: Optional[object] = None) -> str:
    """Format conclusion directly from UniversalResult fields.
    
    Format: {domain} incident — {severity} ({risk}) | Confidence: {confidence}%
    
    Args:
        analysis_result: UniversalResult with all analysis data
        comparison_result: Optional comparison result
        
    Returns:
        Formatted conclusion string
    """
    parts = []
    
    # Header with domain, severity, risk, confidence
    domain = getattr(analysis_result, 'domain', 'unknown')
    confidence = getattr(analysis_result, 'confidence', 0.0)
    
    # Get severity information
    severity_level = "unknown"
    risk_level = "UNKNOWN"
    if hasattr(analysis_result, 'severity') and analysis_result.severity:
        severity_obj = analysis_result.severity
        if hasattr(severity_obj, 'severity'):
            severity_level = severity_obj.severity
            risk_level = getattr(severity_obj, 'risk_level', 'UNKNOWN')
        else:
            severity_level = str(severity_obj)
    
    parts.append(f"{domain.title()} incident — {severity_level.title()} ({risk_level.upper()}) | Confidence: {confidence:.1%}")
    
    # Extract entities and analysis info
    entities, word_count = extract_entities(analysis_result)
    urgency_score, sentiment_label = extract_urgency_sentiment(analysis_result)
    
    # Entity and analysis line
    entity_line = f"{word_count} words"
    if entities:
        entity_line += f". Entities: {', '.join(entities[:5])}"
    parts.append(entity_line)
    
    # Urgency and sentiment line
    parts.append(f"Urgency: {urgency_score:.2f} | Sentiment: {sentiment_label}")
    
    # Add pattern interpretation if available
    if hasattr(analysis_result, 'patterns') and analysis_result.patterns:
        try:
            from iot_machine_learning.infrastructure.ml.cognitive.pattern_interpreter.interpreter import PatternInterpreter
            interpreter = PatternInterpreter()
            pattern_text = interpreter.format_for_conclusion(
                analysis_result.patterns,
                analysis_result.domain,
            )
            parts.append(f"Patrón: {pattern_text}")
        except Exception:
            # Graceful-fail - continue without pattern
            pass
    
    # Add recommended actions
    actions = recommend_actions(analysis_result)
    if actions:
        parts.append("Actions: " + " ".join(actions))
    
    # Add comparative context if available
    if comparison_result and hasattr(comparison_result, 'delta_conclusion'):
        parts.append(f"\n{comparison_result.delta_conclusion}")
    
    # Add Monte Carlo if available
    if hasattr(analysis_result, 'explanation') and analysis_result.explanation:
        explanation_dict = analysis_result.explanation.to_dict()
        if "monte_carlo" in explanation_dict:
            mc = explanation_dict["monte_carlo"]
            if "critical_probability" in mc:
                parts.append(f"Monte Carlo: {mc['critical_probability']:.1%} probability of critical severity")
    
    return "\n".join(parts)


def format_simple_conclusion(analysis_result) -> str:
    """Format simple conclusion without detailed analysis.
    
    Args:
        analysis_result: UniversalResult
        
    Returns:
        Simple conclusion string
    """
    domain = getattr(analysis_result, 'domain', 'unknown')
    confidence = getattr(analysis_result, 'confidence', 0.0)
    
    severity_level = "unknown"
    if hasattr(analysis_result, 'severity') and analysis_result.severity:
        severity_obj = analysis_result.severity
        if hasattr(severity_obj, 'severity'):
            severity_level = severity_obj.severity
        else:
            severity_level = str(severity_obj)
    
    return f"{domain.title()} incident — {severity_level.title()} | Confidence: {confidence:.1%}"
