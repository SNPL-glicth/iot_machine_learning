"""Pure helper functions for severity computation.

Extracted from severity_rules.py to keep the main module under 180 lines.
These are stateless, side-effect-free functions used by both
ThresholdPolicy and the legacy severity_rules interface.
"""

from __future__ import annotations

from typing import Optional


def compute_risk_level_from_threshold(
    value: float,
    threshold: Optional["Threshold"],
) -> str:
    """Compute risk level from a threshold entity."""
    if threshold is None:
        return "normal"
    return threshold.risk_level_for(value)


def action_for_severity(severity: str) -> tuple[bool, str]:
    """Return (action_required, recommended_action) for a severity label."""
    if severity == "critical":
        return True, "Immediate escalation required"
    if severity == "warning":
        return True, "Investigate and monitor closely"
    if severity == "info":
        return False, "Routine monitoring"
    return False, "No action needed"


def severity_from_risk(risk_level: str, out_of_physical_range: bool) -> str:
    """Map risk level + physical range flag to severity label."""
    if out_of_physical_range or risk_level == "critical":
        return "critical"
    if risk_level == "high":
        return "warning"
    if risk_level == "medium":
        return "info"
    return "none"


def build_recommended_action(severity: str) -> str:
    """Build recommended action based on severity."""
    if severity == "critical":
        return "Immediate escalation required - out of physical bounds"
    if severity == "warning":
        return "Investigate and monitor closely"
    if severity == "info":
        return "Routine monitoring recommended"
    return "No action needed"
