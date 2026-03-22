"""Action recommendation based on patterns, severity, and domain."""

from __future__ import annotations

from typing import List, Optional

from iot_machine_learning.infrastructure.ml.cognitive.pattern_interpreter.interpreter import PatternInterpreter


def recommend_actions(analysis_result) -> List[str]:
    """Derive actions from patterns, severity, and domain.
    
    Args:
        analysis_result: UniversalResult with patterns and severity
        
    Returns:
        List of action recommendations
    """
    actions = []
    
    # Get severity level first
    severity_level = "info"
    if hasattr(analysis_result, 'severity') and analysis_result.severity:
        if hasattr(analysis_result.severity, 'severity'):
            severity_level = analysis_result.severity.severity.lower()
    
    # Only get actions that match severity level
    if severity_level == "critical":
        # Only critical actions
        if hasattr(analysis_result, 'patterns') and analysis_result.patterns:
            pattern_actions = _get_critical_pattern_actions(analysis_result.patterns, analysis_result.domain)
            actions.extend(pattern_actions)
        actions.extend(_get_critical_actions())
    elif severity_level == "warning":
        # Only warning actions
        if hasattr(analysis_result, 'patterns') and analysis_result.patterns:
            pattern_actions = _get_warning_pattern_actions(analysis_result.patterns, analysis_result.domain)
            actions.extend(pattern_actions)
        actions.extend(_get_warning_actions())
    else:
        # Only info actions
        actions.extend(_get_info_actions())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_actions = []
    for action in actions:
        if action not in seen:
            seen.add(action)
            unique_actions.append(action)
    
    return unique_actions[:3]  # Limit to top 3 actions


def _get_critical_pattern_actions(patterns: List, domain: str) -> List[str]:
    """Get critical actions from interpreted patterns."""
    actions = []
    
    if not patterns:
        return actions
    
    # Get primary pattern
    interpreter = PatternInterpreter()
    primary_pattern = interpreter.get_primary_pattern(patterns)
    
    if primary_pattern and primary_pattern.severity_hint == "critical":
        if "spike" in primary_pattern.pattern_type:
            actions.append("→ Investigate anomalous spike immediately")
        elif "escalation" in primary_pattern.pattern_type:
            actions.append("→ Escalate incident response team")
        elif "degradation" in primary_pattern.pattern_type:
            actions.append("→ Initiate recovery procedures")
    
    return actions


def _get_warning_pattern_actions(patterns: List, domain: str) -> List[str]:
    """Get warning actions from interpreted patterns."""
    actions = []
    
    if not patterns:
        return actions
    
    # Get primary pattern
    interpreter = PatternInterpreter()
    primary_pattern = interpreter.get_primary_pattern(patterns)
    
    if primary_pattern and primary_pattern.severity_hint == "warning":
        if "drift" in primary_pattern.pattern_type:
            actions.append("→ Monitor drift trend closely")
        elif "regime" in primary_pattern.pattern_type:
            actions.append("→ Verify operational regime change")
        else:
            actions.append("→ Schedule review within 24 hours")
    
    return actions


def _get_critical_actions() -> List[str]:
    """Get critical severity actions."""
    return ["→ Immediate intervention required"]


def _get_warning_actions() -> List[str]:
    """Get warning severity actions."""
    return ["→ Schedule review within 24 hours"]


def _get_info_actions() -> List[str]:
    """Get info severity actions."""
    return ["→ Continue standard monitoring"]


def _get_pattern_actions(patterns: List, domain: str) -> List[str]:
    """Get actions from interpreted patterns."""
    actions = []
    
    if not patterns:
        return actions
    
    # Get primary pattern
    interpreter = PatternInterpreter()
    primary_pattern = interpreter.get_primary_pattern(patterns)
    
    if primary_pattern:
        # Derive actions from pattern severity and type
        if primary_pattern.severity_hint == "critical":
            if "spike" in primary_pattern.pattern_type:
                actions.append("→ Investigate anomalous spike immediately")
            elif "escalation" in primary_pattern.pattern_type:
                actions.append("→ Escalate incident response team")
            elif "degradation" in primary_pattern.pattern_type:
                actions.append("→ Initiate recovery procedures")
        elif primary_pattern.severity_hint == "warning":
            if "drift" in primary_pattern.pattern_type:
                actions.append("→ Monitor drift trend closely")
            elif "regime" in primary_pattern.pattern_type:
                actions.append("→ Verify operational regime change")
            else:
                actions.append("→ Schedule review within 24 hours")
        else:  # info
            actions.append("→ Continue standard monitoring")
    
    return actions


def _get_severity_actions(severity_obj) -> List[str]:
    """Get actions from severity classification."""
    actions = []
    
    if hasattr(severity_obj, 'severity'):
        severity_level = severity_obj.severity.lower()
        
        if severity_level == "critical":
            actions.append("→ Immediate intervention required")
        elif severity_level == "warning":
            actions.append("→ Schedule review within 24 hours")
        elif severity_level == "info":
            actions.append("→ Continue standard monitoring")
    
    return actions
