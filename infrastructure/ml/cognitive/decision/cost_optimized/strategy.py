"""CostOptimizedStrategy — Balances risk reduction with cost efficiency.

Cost-optimized approach: Maximize ROI on interventions.
Defers non-critical actions to off-peak hours when costs are lower.
Uses expected value analysis to justify immediate vs delayed action.
"""

from __future__ import annotations

from typing import Any, Dict

from ......domain.entities.decision import Decision, DecisionContext
from ......domain.ports.decision_port import DecisionEnginePort

from .decision_rules import apply_decision_hierarchy
from .outcome_builder import build_simulated_outcomes


class CostOptimizedStrategy(DecisionEnginePort):
    """Cost-optimized decision strategy.

    Decision hierarchy (first match wins):
    1. severity=critical → escalate (cost irrelevant)
    2. severity=warning AND confidence > 0.7 AND positive ROI → intervene now
    3. severity=warning AND low ROI → schedule (defer to off-peak)
    4. default → monitor (zero cost)

    This strategy:
    - Balances risk reduction with cost efficiency
    - Defers non-critical actions when costs are high
    - Uses ROI calculations to justify immediate vs delayed action
    - Optimizes for total cost of ownership (risk + intervention cost)

    Attributes:
        _version: Strategy version for audit trails
        _confidence_threshold: Minimum confidence for immediate intervene (default 0.7)
        _cost_threshold: Maximum acceptable immediate intervention cost
        _expected_roi: Minimum ROI for immediate action (default 1.5)
        _cost_multiplier: Multiplier for cost estimates
    """

    def __init__(
        self,
        version: str = "1.0.0",
        confidence_threshold: float = 0.7,
        cost_threshold: float = 100.0,
        expected_roi: float = 1.5,
        cost_multiplier: float = 1.0,
    ) -> None:
        """Initialize CostOptimizedStrategy.

        Args:
            version: Version string for audit trails
            confidence_threshold: Confidence threshold for immediate intervene
            cost_threshold: Maximum acceptable immediate cost
            expected_roi: Minimum ROI for immediate action
            cost_multiplier: Adjusts all cost estimates
        """
        self._version = version
        self._confidence_threshold = confidence_threshold
        self._cost_threshold = cost_threshold
        self._expected_roi = expected_roi
        self._cost_multiplier = cost_multiplier

    @property
    def strategy_name(self) -> str:
        """Unique identifier: 'cost_optimized'."""
        return "cost_optimized"

    @property
    def version(self) -> str:
        """Semantic version of this strategy."""
        return self._version

    def can_decide(self, context: DecisionContext) -> bool:
        """Check if context has minimum required fields.

        Cost-optimized can decide with any context (defaults to monitor).

        Args:
            context: Decision context to validate

        Returns:
            True (always can decide)
        """
        return True

    def decide(self, context: DecisionContext) -> Decision:
        """Make cost-optimized decision.

        Applies decision hierarchy with ROI awareness:
        1. Critical conditions always escalate (cost irrelevant)
        2. Warning conditions with positive ROI → intervene now
        3. Warning conditions with low ROI → schedule for off-peak
        4. Default to monitoring (zero cost)

        Args:
            context: Aggregated ML outputs

        Returns:
            Decision with cost-optimized action selection
        """
        # Build simulated outcomes with cost consideration
        simulated = build_simulated_outcomes(context, self._cost_multiplier)

        # Decision hierarchy with cost awareness
        action, priority, reason = apply_decision_hierarchy(
            context,
            simulated,
            self._confidence_threshold,
            self._cost_threshold,
            self._expected_roi,
        )

        # Calculate confidence based on evidence quality and ROI clarity
        confidence = self._calculate_confidence(context, action)

        # Build source ML outputs reference
        source_outputs = self._build_source_outputs(context)

        return Decision(
            action=action,
            priority=priority,
            confidence=confidence,
            reason=reason,
            strategy_used=self.strategy_name,
            simulated_outcomes=simulated,
            source_ml_outputs=source_outputs,
            audit_trace_id=context.audit_trace_id,
        )

    def _calculate_confidence(self, context: DecisionContext, action: str) -> float:
        """Calculate decision confidence.

        Cost-optimized: Confidence based on ROI clarity.
        - Higher confidence when ROI is clearly positive or negative
        - Lower confidence when ROI is near break-even

        Args:
            context: Decision context
            action: Selected action

        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = context.confidence

        # Evidence quality bonus/penalty
        evidence_count = len(context.patterns) + (1 if context.severity else 0)
        if evidence_count >= 3:
            base_confidence = min(0.95, base_confidence + 0.05)
        elif evidence_count == 0:
            base_confidence *= 0.8

        # Action-specific confidence adjustments
        if action == "escalate":
            # Critical escalations have high confidence
            return min(0.98, base_confidence + 0.1)
        elif action == "intervene":
            # Interventions need good confidence (ROI validated)
            return min(0.92, base_confidence)
        elif action == "schedule":
            # Scheduling has moderate confidence (deferral decision)
            return min(0.85, max(0.65, base_confidence * 0.9))
        else:  # monitor
            # Monitoring confidence based on absence of threats
            return min(0.90, max(0.70, 0.85 - base_confidence * 0.2))

    def _build_source_outputs(self, context: DecisionContext) -> Dict[str, Any]:
        """Build reference to source ML outputs.

        Args:
            context: Decision context

        Returns:
            Dictionary of source references
        """
        return {
            "series_id": context.series_id,
            "confidence": context.confidence,
            "is_anomaly": context.is_anomaly,
            "anomaly_score": context.anomaly_score,
            "trend": context.trend,
            "domain": context.domain,
            "pattern_count": len(context.patterns),
            "has_monte_carlo": context.has_monte_carlo,
            "strategy_config": {
                "confidence_threshold": self._confidence_threshold,
                "cost_threshold": self._cost_threshold,
                "expected_roi": self._expected_roi,
                "cost_multiplier": self._cost_multiplier,
            },
        }
