"""Aggressive strategy outcome builder.

Generates optimistic simulated outcomes with best-case analysis.
Uses Monte Carlo if available, otherwise generates optimistic scenarios.
"""

from __future__ import annotations

from typing import List

from ......domain.entities.decision import DecisionContext, SimulatedOutcome


def build_simulated_outcomes(
    context: DecisionContext,
    optimism_factor: float,
) -> List[SimulatedOutcome]:
    """Build simulated outcomes with best-case analysis.

    Aggressive: Optimistic outlook - assumes things will likely go well.
    Reduces risk estimates to justify monitoring over action.

    Args:
        context: Decision context
        optimism_factor: Risk reduction multiplier (e.g., 0.7 = 30% risk reduction)

    Returns:
        List of simulated outcomes
    """
    # Use existing Monte Carlo if available
    if context.monte_carlo_outcomes:
        return context.monte_carlo_outcomes

    # Generate optimistic scenarios based on context
    outcomes = []

    # Scenario 1: Do nothing (optimistic - things resolve naturally)
    base_risk = _estimate_base_risk(context)
    optimistic_risk = base_risk * optimism_factor
    outcomes.append(
        SimulatedOutcome(
            scenario_name="do_nothing",
            probability=0.8,  # Aggressive: 80% chance things resolve
            expected_risk=max(0.0, optimistic_risk),
            description="Optimistic: No action needed, issue likely resolves naturally.",
        )
    )

    # Scenario 2: Act aggressively (minimal improvement assumed)
    # Aggressive: Even intervention doesn't help much (why bother?)
    action_risk = max(0.0, optimistic_risk * 0.9)  # Only 10% better
    outcomes.append(
        SimulatedOutcome(
            scenario_name="act_aggressive",
            probability=0.9,  # High confidence action works IF you do it
            expected_risk=action_risk,
            description="Aggressive action: Minimal risk reduction, high disruption cost.",
        )
    )

    # Scenario 3: Best case (for justifying inaction)
    best_risk = max(0.0, base_risk * optimism_factor * 0.5)  # Very optimistic
    outcomes.append(
        SimulatedOutcome(
            scenario_name="best_case",
            probability=0.6,  # 60% chance of best case
            expected_risk=best_risk,
            description="Best-case scenario with optimistic assumptions.",
        )
    )

    return outcomes


def _estimate_base_risk(context: DecisionContext) -> float:
    """Estimate baseline risk - optimistic version.

    Aggressive risk estimation:
    - Downplays anomaly scores
    - Severity contributes less to risk
    - Patterns only add risk if multiple indicators

    Args:
        context: Decision context

    Returns:
        Risk score between 0.0 and 1.0 (optimistic)
    """
    risk_factors = []

    # Anomaly score contribution (downplayed by 50%)
    if context.is_anomaly and context.anomaly_score > 0:
        risk_factors.append(context.anomaly_score * 0.5)

    # Severity contribution (reduced weights)
    if context.severity:
        severity_risk = {
            "critical": 0.6,  # Reduced from 0.9
            "warning": 0.3,   # Reduced from 0.6
            "info": 0.05,     # Reduced from 0.1
        }.get(context.severity.severity.lower(), 0.3)
        risk_factors.append(severity_risk)

    # Pattern contribution (only count if multiple)
    pattern_count = len(context.patterns)
    for pattern in context.patterns:
        hint = pattern.get("severity_hint", "").lower()
        # Only add significant risk if critical AND multiple patterns
        if hint == "critical" and pattern_count >= 2:
            risk_factors.append(0.5)
        elif hint == "warning":
            risk_factors.append(0.2)
        else:
            risk_factors.append(0.05)

    # Aggressive: take average (optimistic) rather than max (pessimistic)
    if risk_factors:
        return sum(risk_factors) / len(risk_factors)

    return 0.1  # Default low risk when no info
