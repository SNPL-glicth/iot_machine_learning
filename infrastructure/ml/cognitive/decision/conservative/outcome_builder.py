"""Conservative strategy outcome builder.

Generates simulated outcomes with worst-case analysis.
Uses Monte Carlo if available, otherwise generates conservative scenarios.
"""

from __future__ import annotations

from typing import List

from ......domain.entities.decision import DecisionContext, SimulatedOutcome


def build_simulated_outcomes(
    context: DecisionContext,
    safety_margin: float,
) -> List[SimulatedOutcome]:
    """Build simulated outcomes with worst-case analysis.

    If Monte Carlo outcomes available, use them.
    Otherwise, generate conservative scenarios.

    Args:
        context: Decision context
        safety_margin: Multiplier applied to risk in worst-case analysis

    Returns:
        List of simulated outcomes
    """
    # Use existing Monte Carlo if available
    if context.monte_carlo_outcomes:
        return context.monte_carlo_outcomes

    # Generate conservative scenarios based on context
    outcomes = []

    # Scenario 1: Do nothing (baseline risk)
    base_risk = _estimate_base_risk(context)
    outcomes.append(
        SimulatedOutcome(
            scenario_name="do_nothing",
            probability=1.0,
            expected_risk=base_risk,
            description="Baseline: No action taken. Risk evaluated under conservative assumptions.",
        )
    )

    # Scenario 2: Act conservative (reduced risk with safety margin)
    mitigated_risk = max(0.0, base_risk * 0.5)  # 50% risk reduction
    outcomes.append(
        SimulatedOutcome(
            scenario_name="act_conservative",
            probability=min(1.0, context.confidence + 0.1),
            expected_risk=mitigated_risk,
            description="Conservative action: 50% risk reduction assumed with safety margin.",
        )
    )

    # Scenario 3: Worst case (for justification)
    worst_risk = min(1.0, base_risk * safety_margin)
    outcomes.append(
        SimulatedOutcome(
            scenario_name="worst_case",
            probability=0.1,  # Conservative: 10% chance of worst case
            expected_risk=worst_risk,
            description="Worst-case scenario with safety margin applied to risk assessment.",
        )
    )

    return outcomes


def _estimate_base_risk(context: DecisionContext) -> float:
    """Estimate baseline risk from context.

    Conservative risk estimation considers:
    - Anomaly score (if anomaly)
    - Severity level
    - Pattern criticality

    Args:
        context: Decision context

    Returns:
        Risk score between 0.0 and 1.0
    """
    risk_factors = []

    # Anomaly score contribution
    if context.is_anomaly and context.anomaly_score > 0:
        risk_factors.append(context.anomaly_score)

    # Severity contribution
    if context.severity:
        severity_risk = {
            "critical": 0.9,
            "warning": 0.6,
            "info": 0.1,
        }.get(context.severity.severity.lower(), 0.5)
        risk_factors.append(severity_risk)

    # Pattern contribution
    for pattern in context.patterns:
        hint = pattern.get("severity_hint", "").lower()
        pattern_risk = {
            "critical": 0.9,
            "warning": 0.5,
            "info": 0.1,
        }.get(hint, 0.3)
        risk_factors.append(pattern_risk)

    # Conservative: take maximum risk factor (worst case of available evidence)
    # rather than average (which would be optimistic)
    if risk_factors:
        return max(risk_factors)

    return 0.3  # Default moderate risk when no info
