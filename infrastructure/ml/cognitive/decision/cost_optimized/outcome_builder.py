"""Cost-optimized strategy outcome builder.

Generates simulated outcomes with cost-aware analysis.
Uses Monte Carlo if available, otherwise generates ROI-focused scenarios.
"""

from __future__ import annotations

from typing import List

from ......domain.entities.decision import DecisionContext, SimulatedOutcome


def build_simulated_outcomes(
    context: DecisionContext,
    cost_multiplier: float,
) -> List[SimulatedOutcome]:
    """Build simulated outcomes with cost consideration.

    Generates scenarios optimized for ROI calculation:
    - do_nothing: baseline cost + risk
    - intervene_now: immediate cost but reduced risk
    - defer_offpeak: reduced cost but delayed risk reduction

    Args:
        context: Decision context
        cost_multiplier: Multiplier for cost estimates

    Returns:
        List of simulated outcomes with cost data
    """
    # Use existing Monte Carlo if available
    if context.monte_carlo_outcomes:
        return context.monte_carlo_outcomes

    base_risk = _estimate_base_risk(context)
    outcomes = []

    # Scenario 1: Do nothing (zero cost, full risk)
    outcomes.append(
        SimulatedOutcome(
            scenario_name="do_nothing",
            probability=0.5,
            expected_risk=base_risk,
            description=f"No cost. Full risk exposure: {base_risk:.2f}",
        )
    )

    # Scenario 2: Intervene now (immediate cost, 50% risk reduction)
    immediate_cost = 100 * cost_multiplier  # Base cost units
    intervene_risk = base_risk * 0.5
    outcomes.append(
        SimulatedOutcome(
            scenario_name="intervene_now",
            probability=0.8,
            expected_risk=intervene_risk,
            description=f"Cost: {immediate_cost:.0f} units. Risk reduced to {intervene_risk:.2f}",
        )
    )

    # Scenario 3: Defer to off-peak (40% cost reduction, delayed action)
    deferred_cost = immediate_cost * 0.6
    # Risk slightly higher due to delay
    deferred_risk = base_risk * 0.6
    outcomes.append(
        SimulatedOutcome(
            scenario_name="defer_offpeak",
            probability=0.7,
            expected_risk=deferred_risk,
            description=f"Reduced cost: {deferred_cost:.0f} units. Delayed risk: {deferred_risk:.2f}",
        )
    )

    # Scenario 4: Expected value (for ROI calculations)
    # Weighted average based on confidence
    confidence_weight = context.confidence
    expected_risk = (
        base_risk * (1 - confidence_weight) +
        intervene_risk * confidence_weight
    )
    outcomes.append(
        SimulatedOutcome(
            scenario_name="expected_value",
            probability=1.0,
            expected_risk=expected_risk,
            description=f"Confidence-weighted expected risk: {expected_risk:.2f}",
        )
    )

    return outcomes


def _estimate_base_risk(context: DecisionContext) -> float:
    """Estimate baseline risk with moderate weighting.

    Cost-optimized: Balanced risk estimation (not too pessimistic, not too optimistic)

    Args:
        context: Decision context

    Returns:
        Risk score between 0.0 and 1.0
    """
    risk_factors = []
    weights = []

    # Anomaly score contribution
    if context.is_anomaly and context.anomaly_score > 0:
        risk_factors.append(context.anomaly_score)
        weights.append(1.0)

    # Severity contribution
    if context.severity:
        severity_risk = {
            "critical": 0.85,
            "warning": 0.45,
            "info": 0.08,
        }.get(context.severity.severity.lower(), 0.4)
        risk_factors.append(severity_risk)
        weights.append(1.2)  # Severity is important for cost decisions

    # Pattern contribution (weighted by confidence)
    for pattern in context.patterns:
        hint = pattern.get("severity_hint", "").lower()
        pattern_conf = pattern.get("confidence", 0.5)
        pattern_risk = {
            "critical": 0.8,
            "warning": 0.4,
            "info": 0.05,
        }.get(hint, 0.25)
        risk_factors.append(pattern_risk)
        weights.append(pattern_conf)

    # Weighted average if we have factors
    if risk_factors:
        total_weight = sum(weights)
        if total_weight > 0:
            weighted_sum = sum(
                r * w for r, w in zip(risk_factors, weights)
            )
            return weighted_sum / total_weight

    return 0.25  # Default low-moderate risk when no info
