"""Action recommendation based on patterns, severity, and domain."""

from __future__ import annotations

from typing import List, Optional

from .action_catalog import get_actions_for_domain


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
    domain = "general"
    if hasattr(analysis_result, 'severity') and analysis_result.severity:
        if hasattr(analysis_result.severity, 'severity'):
            severity_level = analysis_result.severity.severity.lower()
    
    # Get domain from analysis result
    if hasattr(analysis_result, 'domain') and analysis_result.domain:
        domain = analysis_result.domain.lower()
    
    # Only get actions that match severity level
    if severity_level == "critical":
        # Only critical actions
        if hasattr(analysis_result, 'patterns') and analysis_result.patterns:
            pattern_actions = _get_critical_pattern_actions(analysis_result.patterns, domain)
            actions.extend(pattern_actions)
        actions.extend(_get_critical_actions(domain))
    elif severity_level == "warning":
        # Only warning actions
        if hasattr(analysis_result, 'patterns') and analysis_result.patterns:
            pattern_actions = _get_warning_pattern_actions(analysis_result.patterns, domain)
            actions.extend(pattern_actions)
        actions.extend(_get_warning_actions(domain))
    else:
        # Only info actions
        actions.extend(_get_info_actions(domain))
    
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

    for pattern in patterns:
        severity_hint = getattr(pattern, 'severity_hint', '')
        pattern_type = getattr(pattern, 'pattern_type', '')
        if severity_hint == "critical":
            if "spike" in pattern_type:
                actions.append("→ Investigate anomalous spike immediately")
                break
            elif "escalation" in pattern_type:
                actions.append("→ Escalate incident response team")
                break
            elif "degradation" in pattern_type:
                actions.append("→ Initiate recovery procedures")
                break

    return actions


def _get_warning_pattern_actions(patterns: List, domain: str) -> List[str]:
    """Get warning actions from interpreted patterns."""
    actions = []

    if not patterns:
        return actions

    for pattern in patterns:
        severity_hint = getattr(pattern, 'severity_hint', '')
        pattern_type = getattr(pattern, 'pattern_type', '')
        if severity_hint == "warning":
            if "drift" in pattern_type:
                actions.append("→ Monitor drift trend closely")
            elif "regime" in pattern_type:
                actions.append("→ Verify operational regime change")
            else:
                actions.append("→ Schedule review within 24 hours")
            break

    return actions


def _get_critical_actions(domain: str = "general") -> List[str]:
    """Get critical severity actions for domain."""
    return get_actions_for_domain(domain, "critical", max_actions=3)


def _get_warning_actions(domain: str = "general") -> List[str]:
    """Get warning severity actions for domain."""
    return get_actions_for_domain(domain, "warning", max_actions=3)


def _get_info_actions(domain: str = "general") -> List[str]:
    """Get info severity actions for domain."""
    return get_actions_for_domain(domain, "info", max_actions=2)


def _get_pattern_actions(patterns: List, domain: str) -> List[str]:
    """Get actions from interpreted patterns."""
    if not patterns:
        return []
    return _get_critical_pattern_actions(patterns, domain) or _get_warning_pattern_actions(patterns, domain) or ["→ Continue standard monitoring"]


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
