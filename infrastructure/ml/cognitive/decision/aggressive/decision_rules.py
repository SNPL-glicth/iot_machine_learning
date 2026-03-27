"""Aggressive strategy decision rules.

Implements the decision hierarchy:
1. severity=critical AND confidence > 0.9 → escalate
2. confidence > 0.95 → intervene (high certainty required)
3. default → monitor (prefer inaction)
"""

from __future__ import annotations

from typing import List, Tuple

from ......domain.entities.decision import DecisionContext, SimulatedOutcome
from ......domain.entities.decision.priority import Priority


def apply_decision_hierarchy(
    context: DecisionContext,
    simulated: List[SimulatedOutcome],
    confidence_threshold: float,
    escalation_confidence: float,
) -> Tuple[str, int, str]:
    """Apply aggressive decision hierarchy.

    High bar for action - only act when very certain.
    Prefers false negatives over false positives.

    Args:
        context: Decision context
        simulated: Simulated outcomes for reference
        confidence_threshold: Minimum confidence for intervene
        escalation_confidence: Minimum confidence for escalate

    Returns:
        Tuple of (action, priority, reason)
    """
    # Rule 1: Critical severity AND very high confidence → escalate
    if _is_critical_and_certain(context, escalation_confidence):
        best_risk = _get_best_case_risk(simulated)
        return (
            "escalate",
            Priority.CRITICAL,
            f"Critical condition with {context.confidence:.0%} confidence. "
            f"Best-case risk: {best_risk:.2f}. Escalation warranted.",
        )

    # Rule 2: Very high confidence anomaly → intervene
    if context.confidence > confidence_threshold:
        best_risk = _get_best_case_risk(simulated)
        return (
            "intervene",
            Priority.HIGH,
            f"Very high confidence ({context.confidence:.0%}) anomaly. "
            f"Best-case risk: {best_risk:.2f}. Aggressive: intervene only when certain.",
        )

    # Rule 3: Default → monitor (prefer inaction)
    best_risk = _get_best_case_risk(simulated)
    return (
        "monitor",
        Priority.LOW,
        f"Insufficient certainty ({context.confidence:.0%} < {confidence_threshold:.0%}). "
        f"Best-case risk: {best_risk:.2f}. Aggressive default: monitor. "
        "False negative preferred over disruption.",
    )


def _is_critical_and_certain(context: DecisionContext, threshold: float) -> bool:
    """Check if critical AND highly certain.

    Both conditions must be met:
    - severity.severity == "critical"
    - confidence > threshold

    Args:
        context: Decision context
        threshold: Minimum confidence threshold

    Returns:
        True if critical and certain
    """
    if context.confidence <= threshold:
        return False

    if context.severity:
        sev_str = str(context.severity.severity).lower()
        if sev_str == "critical":
            return True

    return False


def _get_best_case_risk(simulated: List[SimulatedOutcome]) -> float:
    """Get best-case (minimum) risk from simulated outcomes.

    Aggressive: Optimistic risk assessment for justifying inaction.

    Args:
        simulated: List of simulated outcomes

    Returns:
        Minimum risk found
    """
    if not simulated:
        return 0.0  # Aggressive: assume no risk if no data

    return min(o.expected_risk for o in simulated)
