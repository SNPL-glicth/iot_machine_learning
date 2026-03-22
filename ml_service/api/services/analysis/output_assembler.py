"""Output assembly from UniversalResult to final dictionary."""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_output_dict(
    analysis_result,
    comparison_result: Optional[object],
    raw_data: Any,
    semantic_conclusion: Optional[str] = None,
) -> Dict[str, Any]:
    """Build final output dictionary from analysis_result fields."""
    domain = getattr(analysis_result, 'domain', 'unknown')
    confidence = getattr(analysis_result, 'confidence', 0.0)
    input_type = getattr(analysis_result, 'input_type', None)
    
    output = {
        "conclusion": semantic_conclusion or "Analysis completed",
        "domain": domain,
        "severity": _extract_severity_info(analysis_result),
        "confidence": confidence,
        "analysis": {
            "input_type": input_type.value if input_type else 'unknown',
            "full_text": raw_data,
            **_extract_analysis_metadata(analysis_result),
        },
        "explanation": _extract_explanation(analysis_result),
    }
    
    # Add optional components
    if hasattr(analysis_result, 'patterns') and analysis_result.patterns:
        output["patterns"] = [_serialize_pattern(p) for p in analysis_result.patterns]
    
    monte_carlo_data = _extract_monte_carlo(analysis_result)
    if monte_carlo_data:
        output["monte_carlo"] = monte_carlo_data
    
    if hasattr(analysis_result, 'recall_context') and analysis_result.recall_context:
        output["recall_context"] = analysis_result.recall_context
    
    if hasattr(analysis_result, 'pipeline_timing') and analysis_result.pipeline_timing:
        output["pipeline_timing"] = analysis_result.pipeline_timing
    
    return output


def _extract_severity_info(analysis_result) -> Dict[str, Any]:
    """Extract severity information from UniversalResult."""
    if hasattr(analysis_result, 'severity') and analysis_result.severity:
        severity_obj = analysis_result.severity
        if hasattr(severity_obj, 'severity'):
            return {
                "level": severity_obj.severity,
                "risk_level": getattr(severity_obj, 'risk_level', 'UNKNOWN'),
                "action_required": getattr(severity_obj, 'action_required', False),
                "recommended_action": getattr(severity_obj, 'recommended_action', 'Monitor'),
            }
        else:
            return {"level": str(severity_obj), "risk_level": "UNKNOWN", "action_required": False, "recommended_action": "Monitor"}
    return {"level": "unknown", "risk_level": "UNKNOWN", "action_required": False, "recommended_action": "Monitor"}


def _extract_analysis_metadata(analysis_result) -> Dict[str, Any]:
    """Extract analysis metadata from UniversalResult."""
    metadata = {"urgency_score": 0.0, "sentiment_score": 0.0, "word_count": 0, "entities": []}
    
    if hasattr(analysis_result, 'analysis') and analysis_result.analysis:
        analysis = analysis_result.analysis
        metadata.update({k: analysis.get(k, v) for k, v in metadata.items() if k in analysis})
        metadata.update({k: v for k, v in analysis.items() if k not in metadata})
    
    return metadata


def _extract_explanation(analysis_result) -> Dict[str, Any]:
    """Extract explanation from UniversalResult."""
    if hasattr(analysis_result, 'explanation') and analysis_result.explanation:
        return analysis_result.explanation.to_dict()
    return {"note": "Detailed explanation not available"}


def _serialize_pattern(pattern) -> Dict[str, Any]:
    """Serialize InterpretedPattern to dictionary."""
    if hasattr(pattern, 'to_dict'):
        return pattern.to_dict()
    
    return {
        "pattern_type": getattr(pattern, 'pattern_type', 'unknown'),
        "short_name": getattr(pattern, 'short_name', 'Unknown'),
        "description": getattr(pattern, 'description', ''),
        "severity_hint": getattr(pattern, 'severity_hint', 'info'),
        "domain_context": getattr(pattern, 'domain_context', ''),
        "confidence": getattr(pattern, 'confidence', 0.0),
        "data_type": getattr(pattern, 'data_type', 'unknown'),
    }


def _extract_monte_carlo(analysis_result) -> Optional[Dict[str, Any]]:
    """Extract Monte Carlo data from UniversalResult."""
    if hasattr(analysis_result, 'monte_carlo') and analysis_result.monte_carlo:
        mc = analysis_result.monte_carlo
        return {
            "severity_distribution": getattr(mc, 'severity_distribution', {}),
            "confidence_interval": getattr(mc, 'confidence_interval', [0.0, 1.0]),
            "expected_severity": getattr(mc, 'expected_severity', 'unknown'),
            "confidence_score": getattr(mc, 'confidence_score', 0.0),
            "uncertainty_level": getattr(mc, 'uncertainty_level', 'medium'),
            "scenarios": getattr(mc, 'scenario_outcomes', []),
        }
    return None
