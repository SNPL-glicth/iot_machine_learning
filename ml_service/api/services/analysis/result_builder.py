"""Result builder for document analysis.

Builds human-readable conclusions and structured output dictionaries.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_conclusion(
    analysis_result,
    comparison_result: Optional[object],
) -> str:
    """Build human-readable conclusion from analysis + comparison.
    
    Args:
        analysis_result: UniversalAnalysisResult from engine
        comparison_result: Optional comparison result
        
    Returns:
        Multi-line conclusion string
    """
    parts = []
    
    # Main analysis conclusion from explanation
    explanation_dict = analysis_result.explanation.to_dict()
    if "reasoning" in explanation_dict:
        reasoning = explanation_dict["reasoning"]
        if "phases" in reasoning and reasoning["phases"]:
            # Extract phase conclusions
            for phase in reasoning["phases"]:
                if "result" in phase and phase["result"]:
                    parts.append(f"• {phase['result']}")
    
    # Add severity assessment
    severity = analysis_result.severity
    parts.append(f"\nSeverity: {severity.severity} (Risk: {severity.risk_level})")
    if severity.action_required:
        parts.append(f"Action: {severity.recommended_action}")
    
    # Add comparative context if available
    if comparison_result:
        parts.append(f"\n{comparison_result.delta_conclusion}")
    
    return "\n".join(parts) if parts else "Analysis completed successfully."


def build_output_dict(
    analysis_result,
    comparison_result: Optional[object],
    raw_data: Any,
) -> Dict[str, Any]:
    """Build structured output dictionary from analysis results.
    
    Args:
        analysis_result: UniversalAnalysisResult from engine
        comparison_result: Optional comparison result
        raw_data: Original raw data
        
    Returns:
        Structured output dictionary
    """
    # Build response
    conclusion = build_conclusion(analysis_result, comparison_result)
    output = {
        "conclusion": conclusion,
        "domain": analysis_result.domain,
        "severity": {
            "level": analysis_result.severity.severity,
            "risk_level": analysis_result.severity.risk_level,
            "action_required": analysis_result.severity.action_required,
            "recommended_action": analysis_result.severity.recommended_action,
        },
        "confidence": analysis_result.confidence,
        "analysis": {
            "input_type": analysis_result.input_type.value,
            "full_text": raw_data,
            "urgency_score": analysis_result.analysis.get("urgency_score", 0),
            "sentiment_score": analysis_result.analysis.get("sentiment_score", 0),
            **analysis_result.analysis,
        },
        "explanation": analysis_result.explanation.to_dict(),
    }
    
    # Include Monte Carlo result if available
    if analysis_result.monte_carlo is not None:
        output["monte_carlo"] = {
            "severity_distribution": analysis_result.monte_carlo.severity_distribution,
            "confidence_interval": {
                "lower": analysis_result.monte_carlo.confidence_interval[0],
                "upper": analysis_result.monte_carlo.confidence_interval[1],
            },
            "expected_severity": analysis_result.monte_carlo.expected_severity,
            "confidence_score": analysis_result.monte_carlo.confidence_score,
            "uncertainty_level": analysis_result.monte_carlo.uncertainty_level,
            "scenarios": analysis_result.monte_carlo.scenario_outcomes,
        }

    return output
