"""Conservative strategy decision rules.

Implements the decision hierarchy:
1. severity=critical OR pattern contains critical → escalate
2. confidence > threshold → intervene
3. default → investigate
"""

from __future__ import annotations

from typing import List, Tuple

from ......domain.entities.decision import DecisionContext, SimulatedOutcome
from ......domain.entities.decision.priority import Priority


def apply_decision_hierarchy(
    context: DecisionContext,
    simulated: List[SimulatedOutcome],
    confidence_threshold: float,
) -> Tuple[str, int, str]:
    """Apply conservative decision hierarchy.

    Args:
        context: Decision context
        simulated: Simulated outcomes for reference
        confidence_threshold: Minimum confidence for intervene action

    Returns:
        Tuple of (action, priority, reason)
    """
    # Rule 1: Critical severity OR escalation pattern → escalate
    if _is_critical_condition(context):
        worst_risk = _get_worst_case_risk(simulated)
        return (
            "escalate",
            Priority.CRITICAL,
            f"Critical condition detected. Worst-case risk: {worst_risk:.2f}. "
            f"Immediate escalation required per conservative policy.",
        )

    # Rule 2: High confidence anomaly → intervene
    if context.confidence > confidence_threshold:
        worst_risk = _get_worst_case_risk(simulated)
        return (
            "intervene",
            Priority.HIGH,
            f"High confidence ({context.confidence:.2f}) anomaly detected. "
            f"Worst-case risk: {worst_risk:.2f}. Conservative action: intervene.",
        )

    # Rule 3: Default → investigate (better safe than sorry)
    return (
        "investigate",
        Priority.MEDIUM,
        "Uncertain conditions detected. Conservative default: investigate "
        "to rule out potential issues. False positive preferred over miss.",
    )


def _is_critical_condition(context: DecisionContext) -> bool:
    """Check if context indicates critical condition.

    Critical conditions:
    - severity.severity == "critical"
    - Any pattern has severity_hint == "critical"
    - High anomaly score (>0.8) with action required

    Args:
        context: Decision context

    Returns:
        True if critical condition detected
    """
    # Check severity
    if context.severity:
        sev_str = str(context.severity.severity).lower()
        if sev_str == "critical":
            return True
        if context.severity.action_required and context.anomaly_score > 0.8:
            return True

    # Check patterns for escalation indicators
    for pattern in context.patterns:
        hint = pattern.get("severity_hint", "").lower()
        if hint == "critical":
            return True

    return False


def _get_worst_case_risk(simulated: List[SimulatedOutcome]) -> float:
    """Get worst-case risk from simulated outcomes.

    Args:
        simulated: List of simulated outcomes

    Returns:
        Maximum risk found
    """
    if not simulated:
        return 1.0  # Conservative: assume max risk if no data

    return max(o.expected_risk for o in simulated)
