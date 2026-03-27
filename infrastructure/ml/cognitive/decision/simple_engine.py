"""SimpleDecisionEngine — MVP passthrough decision engine.

Minimal implementation that maps severity directly to decisions.
No complex strategy logic, no Monte Carlo simulation.
Just passes through the severity result as a decision.

This is the default engine when Decision Engine is enabled.
Guarantees zero functional changes to existing pipeline.
"""

from __future__ import annotations

from typing import Optional

from .....domain.entities.decision import Decision, DecisionContext
from .....domain.ports.decision_port import DecisionEnginePort


class SimpleDecisionEngine(DecisionEnginePort):
    """MVP passthrough decision engine.

    Maps severity results directly to decisions without strategy logic.
    This is the safest default: it makes decisions visible but doesn't
    change the underlying ML pipeline behavior.

    Decision mapping:
    - severity=critical → action=escalate, priority=1
    - severity=warning → action=investigate, priority=2
    - severity=info → action=monitor, priority=4

    Attributes:
        _version: Engine version for audit trails
    """

    def __init__(self, version: str = "1.0.0") -> None:
        """Initialize SimpleDecisionEngine.

        Args:
            version: Version string for audit trails (default "1.0.0")
        """
        self._version = version

    @property
    def strategy_name(self) -> str:
        """Unique identifier: 'simple'."""
        return "simple"

    @property
    def version(self) -> str:
        """Semantic version of this engine."""
        return self._version

    def can_decide(self, context: DecisionContext) -> bool:
        """Check if context has required fields.

        Simple engine only requires severity to be present.
        All other fields are optional.

        Args:
            context: Decision context to validate

        Returns:
            True (always can decide with severity present)
        """
        # SeverityResult is always present (has default_factory)
        # No complex validation needed for passthrough
        return context.severity is not None

    def decide(self, context: DecisionContext) -> Decision:
        """Make decision by mapping severity directly.

        MVP passthrough: uses Decision.from_severity() factory
        to map severity → action/priority without strategy logic.

        Args:
            context: Aggregated ML outputs

        Returns:
            Decision with action derived from severity
        """
        # Use factory method for clean passthrough
        decision = Decision.from_severity(
            severity=context.severity,
            series_id=context.series_id,
            strategy=self.strategy_name,
        )

        # Enrich with audit trace ID if available
        if context.audit_trace_id and not decision.audit_trace_id:
            # Note: Decision is frozen, so we create new instance
            decision = Decision(
                action=decision.action,
                priority=decision.priority,
                confidence=decision.confidence,
                reason=decision.reason,
                strategy_used=decision.strategy_used,
                simulated_outcomes=decision.simulated_outcomes,
                source_ml_outputs={
                    **decision.source_ml_outputs,
                    "audit_trace_id": context.audit_trace_id,
                },
                audit_trace_id=context.audit_trace_id,
            )

        return decision
