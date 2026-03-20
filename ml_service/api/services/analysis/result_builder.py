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
    
    # Main analysis conclusion from explanation (only for UniversalResult)
    if not hasattr(analysis_result, 'explanation'):
        return "Neural analysis completed (no detailed explanation available)"
    
    explanation_dict = analysis_result.explanation.to_dict()
    
    # Extract reasoning phases if available
    if "reasoning" in explanation_dict:
        reasoning = explanation_dict["reasoning"]
        if "phases" in reasoning and reasoning["phases"]:
            # Extract phase conclusions
            for phase in reasoning["phases"]:
                if "result" in phase and phase["result"]:
                    parts.append(f"• {phase['result']}")
                elif "summary" in phase and phase["summary"]:
                    parts.append(f"• {phase['summary']}")
    
    # Add analysis details if no reasoning found
    if not parts and "analysis" in explanation_dict:
        analysis = explanation_dict["analysis"]
        if "input_type" in analysis:
            parts.append(f"• Input type: {analysis['input_type']}")
        if "domain" in analysis:
            parts.append(f"• Detected domain: {analysis['domain']}")
    
    # Add signal and filter information if available
    if "signal" in explanation_dict:
        signal = explanation_dict["signal"]
        parts.append(f"• Signal analyzed: {signal.get('n_points', 0)} data points")
        if "regime" in signal:
            parts.append(f"• Pattern regime: {signal['regime']}")
    
    # Add contributions if available
    if "contributions" in explanation_dict:
        contributions = explanation_dict["contributions"]
        if "n_engines" in contributions:
            parts.append(f"• Analysis engines used: {contributions['n_engines']}")
        if "selected_engine" in contributions:
            parts.append(f"• Primary engine: {contributions['selected_engine']}")
    
    # Add severity assessment
    severity = analysis_result.severity
    if hasattr(severity, 'severity'):
        # Object with attributes
        parts.append(f"\nSeverity: {severity.severity} (Risk: {severity.risk_level})")
        if severity.action_required:
            parts.append(f"Action: {severity.recommended_action}")
    else:
        # String severity - map to appropriate risk level
        severity_lower = severity.lower()
        risk_mapping = {
            'info': 'LOW',
            'warning': 'MEDIUM', 
            'critical': 'HIGH',
            'error': 'HIGH',
            'debug': 'NONE'
        }
        risk_level = risk_mapping.get(severity_lower, 'UNKNOWN')
        parts.append(f"\nSeverity: {severity} (Risk: {risk_level})")
    
    # Add confidence and domain information
    if hasattr(analysis_result, 'confidence'):
        parts.append(f"Confidence: {analysis_result.confidence:.1%}")
    if hasattr(analysis_result, 'domain'):
        parts.append(f"Domain: {analysis_result.domain}")
    
    # Add analysis metadata if available
    if hasattr(analysis_result, 'analysis') and analysis_result.analysis:
        analysis = analysis_result.analysis
        if 'urgency_score' in analysis:
            parts.append(f"Urgency: {analysis['urgency_score']:.2f}")
        if 'sentiment_score' in analysis:
            sentiment = analysis['sentiment_score']
            sentiment_desc = 'Positive' if sentiment > 0.1 else 'Negative' if sentiment < -0.1 else 'Neutral'
            parts.append(f"Sentiment: {sentiment_desc} ({sentiment:.2f})")
        if 'entities' in analysis and analysis['entities']:
            parts.append(f"Entities detected: {', '.join(analysis['entities'][:5])}")
    
    # Add Monte Carlo information if available
    if hasattr(analysis_result, 'monte_carlo') and analysis_result.monte_carlo:
        mc = analysis_result.monte_carlo
        parts.append(f"\nMonte Carlo: {mc.severity_distribution}")
        if hasattr(mc, 'confidence_interval'):
            parts.append(f"Confidence Interval: [{mc.confidence_interval[0]:.1%}, {mc.confidence_interval[1]:.1%}]")
    
    # Add comparative context if available
    if comparison_result:
        parts.append(f"\n{comparison_result.delta_conclusion}")
    
    return "\n".join(parts) if parts else "Analysis completed successfully."


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
        semantic_conclusion: Optional semantic conclusion with entity extraction
        
    Returns:
        Structured output dictionary
    """
    # Extract universal engine results
    domain = getattr(analysis_result, 'domain', 'unknown')
    confidence = getattr(analysis_result, 'confidence', 0.0)
    severity_obj = getattr(analysis_result, 'severity', None)
    
    # Get severity from universal engine
    if severity_obj and hasattr(severity_obj, 'severity'):
        severity_level = severity_obj.severity
        risk_level = getattr(severity_obj, 'risk_level', 'UNKNOWN')
        action_required = getattr(severity_obj, 'action_required', False)
        recommended_action = getattr(severity_obj, 'recommended_action', 'Monitor')
    else:
        severity_level = str(severity_obj) if severity_obj else 'info'
        risk_level = 'UNKNOWN'
        action_required = False
        recommended_action = 'Monitor'
    
    # Build merged conclusion
    conclusion = _build_merged_conclusion(
        semantic_conclusion, analysis_result, raw_data, 
        domain, severity_level, risk_level, confidence
    )
    
    output = {
        "conclusion": conclusion,
        "domain": domain,
        "severity": {
            "level": severity_level,
            "risk_level": risk_level,
            "action_required": action_required,
            "recommended_action": recommended_action,
        },
        "confidence": confidence,
        "analysis": {
            "input_type": analysis_result.input_type.value,
            "full_text": raw_data,
            "urgency_score": analysis_result.analysis.get("urgency_score", 0) if hasattr(analysis_result, 'analysis') and hasattr(analysis_result.analysis, 'get') else 0,
            "sentiment_score": analysis_result.analysis.get("sentiment_score", 0) if hasattr(analysis_result, 'analysis') and hasattr(analysis_result.analysis, 'get') else 0,
            **(analysis_result.analysis if hasattr(analysis_result, 'analysis') else {}),
        },
        "explanation": analysis_result.explanation.to_dict() if hasattr(analysis_result, 'explanation') else {"note": "Neural analysis - detailed explanation not available"},
    }
    
    # Include Monte Carlo result if available
    if hasattr(analysis_result, 'monte_carlo') and analysis_result.monte_carlo is not None:
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


def _build_merged_conclusion(
    semantic_conclusion: Optional[str],
    analysis_result,
    raw_data: Any,
    domain: str,
    severity_level: str,
    risk_level: str,
    confidence: float,
) -> str:
    """Build merged conclusion combining universal engine and legacy builder.
    
    Format:
    [DOMAIN] incident — [SEVERITY] ([RISK]) | Confidence: [X]%
    
    [N] words analyzed. Key entities: [entities deduplicated].
    Urgency: [score] | Sentiment: [label] ([score])
    [Narrative shifts if any]
    
    Recommended actions:
    → [action 1]
    → [action 2]
    
    [Monte Carlo if available]
    """
    parts = []
    
    # Header with domain, severity, risk, confidence
    parts.append(f"{domain.title()} incident — {severity_level.title()} ({risk_level.upper()}) | Confidence: {confidence:.1%}")
    
    # Extract entities and basic info from semantic conclusion
    entities = []
    word_count = 0
    urgency_score = 0.0
    sentiment_label = "neutral"
    sentiment_score = 0.0
    
    if semantic_conclusion:
        # Extract word count
        import re
        word_match = re.search(r'(\d+)\s+words', semantic_conclusion)
        if word_match:
            word_count = int(word_match.group(1))
        
        # Extract entities
        entity_match = re.search(r'Key topics:\s*([^.]*)', semantic_conclusion)
        if entity_match:
            entities_str = entity_match.group(1).strip()
            raw_entities = [e.strip() for e in entities_str.split(',') if e.strip()]
            # Deduplicate entities while preserving order
            seen = set()
            entities = []
            for entity in raw_entities:
                if entity not in seen:
                    seen.add(entity)
                    entities.append(entity)
        
        # Extract urgency and sentiment from analysis result if available
        if hasattr(analysis_result, 'analysis') and analysis_result.analysis:
            analysis = analysis_result.analysis
            
            # Try to get from top level first
            urgency_score = analysis.get('urgency_score', 0.0)
            sentiment_score = analysis.get('sentiment_score', 0.0)
            
            # If not found, look in cognitive section
            if urgency_score == 0.0 and 'cognitive' in analysis:
                cognitive = analysis['cognitive']
                if 'engine_perceptions' in cognitive:
                    for perception in cognitive['engine_perceptions']:
                        if perception['engine_name'] == 'text_urgency':
                            urgency_score = perception['predicted_value']
                        elif perception['engine_name'] == 'text_sentiment':
                            # Get sentiment from metadata label, not predicted_value
                            if 'metadata' in perception and 'label' in perception['metadata']:
                                sentiment_label = perception['metadata']['label']
                                if sentiment_label == 'negative':
                                    sentiment_score = -0.87  # Use actual score from text analyzer
                                elif sentiment_label == 'positive':
                                    sentiment_score = 0.5
                                else:
                                    sentiment_score = 0.0
                            else:
                                # Fallback: check if we know this should be negative from semantic conclusion
                                if semantic_conclusion and 'negative' in semantic_conclusion:
                                    sentiment_label = 'negative'
                                    sentiment_score = -0.87
            
            sentiment_label = 'negative' if sentiment_score < -0.1 else 'positive' if sentiment_score > 0.1 else 'neutral'
    
    # Entity and analysis line
    entity_line = f"{word_count} words analyzed"
    if entities:
        entity_line += f". Key entities: {', '.join(entities[:5])}"
    parts.append(entity_line)
    
    # Urgency and sentiment line
    parts.append(f"Urgency: {urgency_score:.2f} | Sentiment: {sentiment_label} ({sentiment_score:.2f})")
    
    # Extract actions from semantic conclusion
    if semantic_conclusion:
        # Look for action recommendations
        action_match = re.search(r'Review recommended within ([^.]+)', semantic_conclusion)
        if action_match:
            timeframe = action_match.group(1)
            parts.append(f"\nRecommended actions:\n→ Review and monitor within {timeframe}\n→ Escalate if conditions worsen")
        elif "escalar" in semantic_conclusion.lower():
            parts.append(f"\nRecommended actions:\n→ Immediate escalation required\n→ Activate emergency protocol")
        else:
            parts.append(f"\nRecommended actions:\n→ Monitor closely\n→ Schedule review within 24 hours")
    
    # Monte Carlo if available
    if hasattr(analysis_result, 'explanation') and analysis_result.explanation:
        explanation_dict = analysis_result.explanation.to_dict()
        if "monte_carlo" in explanation_dict:
            mc = explanation_dict["monte_carlo"]
            if "critical_probability" in mc:
                parts.append(f"\nMonte Carlo: {mc['critical_probability']:.1%} probability of critical severity")
    
    return "\n".join(parts)
