"""AggressiveStrategy — Prefers false negatives over false positives.

Aggressive approach: Only act when very certain.
High bar for action - prefers inaction over potential disruption.
Uses best-case Monte Carlo scenarios to justify monitoring.
"""

from __future__ import annotations

from typing import Any, Dict

from ......domain.entities.decision import Decision, DecisionContext
from ......domain.ports.decision_port import DecisionEnginePort

from .decision_rules import apply_decision_hierarchy
from .outcome_builder import build_simulated_outcomes


class AggressiveStrategy(DecisionEnginePort):
    """Aggressive decision strategy.

    Decision hierarchy (first match wins):
    1. severity=critical AND confidence > 0.9 → escalate
    2. confidence > 0.95 → intervene (very high certainty)
    3. default → monitor (prefer inaction)

    This strategy:
    - Prefers false negatives over false positives (don't disrupt unnecessarily)
    - Uses Monte Carlo best-case scenario to justify monitoring
    - Only considers very high-confidence anomalies as actionable
    - Reduces risk estimates optimistically

    Attributes:
        _version: Strategy version for audit trails
        _confidence_threshold: Minimum confidence for "intervene" (default 0.95)
        _escalation_confidence: Minimum confidence for "escalate" (default 0.9)
        _optimism_factor: Risk reduction multiplier (default 0.7)
    """

    def __init__(
        self,
        version: str = "1.0.0",
        confidence_threshold: float = 0.95,
        escalation_confidence: float = 0.9,
        optimism_factor: float = 0.7,
    ) -> None:
        """Initialize AggressiveStrategy.

        Args:
            version: Version string for audit trails
            confidence_threshold: Very high threshold for "intervene" action
            escalation_confidence: High threshold for "escalate" (with critical)
            optimism_factor: Multiplier that reduces risk estimates
        """
        self._version = version
        self._confidence_threshold = confidence_threshold
        self._escalation_confidence = escalation_confidence
        self._optimism_factor = optimism_factor

    @property
    def strategy_name(self) -> str:
        """Unique identifier: 'aggressive'."""
        return "aggressive"

    @property
    def version(self) -> str:
        """Semantic version of this strategy."""
        return self._version

    def can_decide(self, context: DecisionContext) -> bool:
        """Check if context has minimum required fields.

        Aggressive strategy is more lenient - can always default to monitor.

        Args:
            context: Decision context to validate

        Returns:
            True (always can decide, even if just to monitor)
        """
        return True  # Aggressive: can always decide to monitor

    def decide(self, context: DecisionContext) -> Decision:
        """Make aggressive decision.

        Applies decision hierarchy:
        1. Check for critical conditions with very high confidence
        2. Check for extremely high-confidence anomalies
        3. Default to monitoring (prefer inaction)

        Args:
            context: Aggregated ML outputs

        Returns:
            Decision with aggressive action selection
        """
        # Build simulated outcomes with optimistic analysis
        simulated = build_simulated_outcomes(context, self._optimism_factor)

        # Decision hierarchy: escalate (rare) → intervene (rare) → monitor (default)
        action, priority, reason = apply_decision_hierarchy(
            context,
            simulated,
            self._confidence_threshold,
            self._escalation_confidence,
        )

        # Calculate confidence (aggressive: higher confidence in inaction)
        confidence = self._calculate_confidence(context)

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

    def _calculate_confidence(self, context: DecisionContext) -> float:
        """Calculate decision confidence.

        Aggressive confidence calculation:
        - Boosted when recommending monitor (we're confident in inaction)
        - Reduced when recommending action (need to be very sure)
        - Range: 0.7 to 0.98

        Args:
            context: Decision context

        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = context.confidence

        # Aggressive boost: we're very confident when there's no clear threat
        if base_confidence < self._confidence_threshold:
            return min(0.95, max(0.75, 0.85 + (1 - base_confidence) * 0.1))

        # For action recommendations, need very high certainty
        return min(0.98, max(0.7, base_confidence))

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
                "escalation_confidence": self._escalation_confidence,
                "optimism_factor": self._optimism_factor,
            },
        }
