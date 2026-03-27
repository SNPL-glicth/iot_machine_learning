"""Cost-optimized strategy decision rules.

Implements the decision hierarchy with cost awareness:
1. severity=critical → escalate (cost irrelevant for critical)
2. severity=warning AND confidence > 0.7 AND cost < threshold → intervene
3. severity=warning AND cost > threshold → schedule (defer to off-peak)
4. default → monitor
"""

from __future__ import annotations

from typing import List, Tuple, Optional

from ......domain.entities.decision import DecisionContext, SimulatedOutcome
from ......domain.entities.decision.priority import Priority


def apply_decision_hierarchy(
    context: DecisionContext,
    simulated: List[SimulatedOutcome],
    confidence_threshold: float,
    cost_threshold: float,
    expected_roi: float,
) -> Tuple[str, int, str]:
    """Apply cost-optimized decision hierarchy.

    Balances risk reduction with cost efficiency.
    Defers non-critical actions when costs are high.

    Args:
        context: Decision context
        simulated: Simulated outcomes for ROI calculation
        confidence_threshold: Minimum confidence for intervene
        cost_threshold: Maximum acceptable immediate cost
        expected_roi: Minimum ROI for immediate action

    Returns:
        Tuple of (action, priority, reason)
    """
    # Calculate expected value from simulated outcomes
    expected_risk = _calculate_expected_value(simulated)
    
    # Rule 1: Critical severity → escalate (cost irrelevant for critical)
    if _is_critical(context):
        return (
            "escalate",
            Priority.CRITICAL,
            f"Critical condition. Expected risk: {expected_risk:.2f}. "
            f"Immediate escalation required regardless of cost.",
        )

    # Rule 2: Warning with good confidence AND positive ROI → intervene now
    if _is_warning(context) and context.confidence > confidence_threshold:
        roi = _calculate_roi(expected_risk, cost_threshold)
        if roi > expected_roi:
            return (
                "intervene",
                Priority.HIGH,
                f"Warning condition with {context.confidence:.0%} confidence. "
                f"Expected risk: {expected_risk:.2f}, ROI: {roi:.1f}x. "
                f"Cost-effective to intervene now.",
            )
        else:
            # Positive expected value but low ROI - schedule instead
            return (
                "schedule",
                Priority.MEDIUM,
                f"Warning detected but ROI ({roi:.1f}x) below threshold. "
                f"Expected risk: {expected_risk:.2f}. Schedule for off-peak "
                f"to reduce cost from {cost_threshold:.0f} units.",
            )

    # Rule 3: Low confidence warning → schedule for later review
    if _is_warning(context):
        return (
            "schedule",
            Priority.MEDIUM,
            f"Uncertain warning ({context.confidence:.0%} confidence). "
            f"Expected risk: {expected_risk:.2f}. Schedule for next review cycle.",
        )

    # Rule 4: Default → monitor (no immediate cost)
    return (
        "monitor",
        Priority.LOW,
        f"No actionable conditions. Expected risk: {expected_risk:.2f}. "
        f"Continue monitoring at zero cost.",
    )


def _is_critical(context: DecisionContext) -> bool:
    """Check if severity is critical.

    Args:
        context: Decision context

    Returns:
        True if severity is critical
    """
    if context.severity:
        return str(context.severity.severity).lower() == "critical"
    return False


def _is_warning(context: DecisionContext) -> bool:
    """Check if severity is warning.

    Args:
        context: Decision context

    Returns:
        True if severity is warning
    """
    if context.severity:
        return str(context.severity.severity).lower() == "warning"
    return False


def _calculate_expected_value(simulated: List[SimulatedOutcome]) -> float:
    """Calculate expected value (risk) from simulated outcomes.

    Uses probability-weighted average of all scenarios.

    Args:
        simulated: List of simulated outcomes

    Returns:
        Expected risk value
    """
    if not simulated:
        return 0.5  # Neutral expectation if no data

    total_prob = sum(o.probability for o in simulated)
    if total_prob == 0:
        return 0.5

    weighted_risk = sum(o.expected_risk * o.probability for o in simulated)
    return weighted_risk / total_prob


def _calculate_roi(expected_risk: float, cost: float) -> float:
    """Calculate return on investment for intervention.

    ROI = (Risk reduction value) / (Intervention cost)
    Assumes intervention reduces risk by 50%.

    Args:
        expected_risk: Expected risk without intervention
        cost: Cost of intervention

    Returns:
        ROI ratio (1.0 = break even, 2.0 = 100% return)
    """
    if cost <= 0:
        return float("inf")  # Free action = infinite ROI

    risk_reduction = expected_risk * 0.5  # Assume 50% risk reduction
    value_of_reduction = risk_reduction * 100  # Risk value in cost units

    return value_of_reduction / cost
