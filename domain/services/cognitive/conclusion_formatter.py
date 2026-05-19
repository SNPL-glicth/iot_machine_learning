"""Conclusion formatting from UniversalResult data.

Domain-layer module: no infrastructure or framework dependencies.
Callers are responsible for extracting entities before calling format_conclusion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..actions.action_recommender import recommend_actions


def format_conclusion(
    analysis_result,
    entities: Dict[str, Any],
    comparison_result: Optional[object] = None,
) -> str:
    """Format conclusion directly from UniversalResult fields.

    Format: {domain} incident — {severity} ({risk}) | Confidence: {confidence}%

    Args:
        analysis_result: UniversalResult with all analysis data.
        entities: Pre-extracted entity dict with keys:
            - entities: List[str]
            - word_count: int
            - urgency_score: float
            - sentiment_label: str
        comparison_result: Optional comparison result.

    Returns:
        Formatted conclusion string.
    """
    
    parts = []
    
    # Header with domain, severity, risk, confidence
    # Handle both UniversalResult and AnalysisResult wrapper
    domain = 'unknown'
    confidence = 0.0
    
    # Try direct access first (UniversalResult)
    if hasattr(analysis_result, 'domain'):
        domain = analysis_result.domain
    # Try signal.domain (AnalysisResult wrapper)
    elif hasattr(analysis_result, 'signal') and analysis_result.signal:
        domain = getattr(analysis_result.signal, 'domain', 'unknown')
    
    # Try direct access first (UniversalResult)
    if hasattr(analysis_result, 'confidence'):
        confidence = analysis_result.confidence
    # Try decision.confidence (AnalysisResult wrapper)
    elif hasattr(analysis_result, 'decision') and analysis_result.decision:
        confidence = getattr(analysis_result.decision, 'confidence', 0.0)
    
    # Get severity information
    # Handle both UniversalResult and AnalysisResult wrapper
    severity_level = "unknown"
    risk_level = "UNKNOWN"
    
    severity_obj = None
    # Try direct access first (UniversalResult)
    if hasattr(analysis_result, 'severity') and analysis_result.severity:
        severity_obj = analysis_result.severity
    # Try decision.severity (AnalysisResult wrapper)
    elif hasattr(analysis_result, 'decision') and analysis_result.decision:
        severity_obj = getattr(analysis_result.decision, 'severity', None)
    
    if severity_obj:
        if hasattr(severity_obj, 'severity'):
            severity_level = severity_obj.severity
            risk_level = getattr(severity_obj, 'risk_level', 'UNKNOWN')
        else:
            severity_level = str(severity_obj)
    
    parts.append(f"{domain.title()} incident — {severity_level.title()} ({risk_level.upper()}) | Confidence: {confidence:.1%}")
    
    # Entity and analysis line (from pre-extracted entities dict)
    word_count: int = entities.get("word_count", 0)
    entity_list: List[str] = entities.get("entities", [])
    urgency_score: float = entities.get("urgency_score", 0.0)
    sentiment_label: str = entities.get("sentiment_label", "neutral")

    entity_line = f"{word_count} words"
    if entity_list:
        entity_line += f". Entities: {', '.join(entity_list[:5])}"
    parts.append(entity_line)

    parts.append(f"Urgency: {urgency_score:.2f} | Sentiment: {sentiment_label}")
    
    # Add pattern interpretation if pre-computed by caller
    pattern_text: Optional[str] = entities.get("pattern_text")
    if pattern_text:
        parts.append(f"Patrón: {pattern_text}")

    # Add recommended actions
    actions = recommend_actions(analysis_result)
    if actions:
        parts.append("Actions: " + " ".join(actions))
    
    # Add comparative context if available
    if comparison_result and hasattr(comparison_result, 'delta_conclusion'):
        parts.append(f"\n{comparison_result.delta_conclusion}")
    
    # Add Monte Carlo if available
    if hasattr(analysis_result, 'explanation') and analysis_result.explanation:
        # Handle both Explanation types (with and without to_dict)
        explanation_dict = None
        if hasattr(analysis_result.explanation, 'to_dict'):
            try:
                explanation_dict = analysis_result.explanation.to_dict()
            except Exception:
                pass
        
        if explanation_dict and "monte_carlo" in explanation_dict:
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
