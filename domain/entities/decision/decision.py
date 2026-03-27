"""Decision — recommended action with full explainability.

Immutable value object representing a decision recommendation.
Contains all information needed to understand, audit, and act.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ....domain.services.severity_rules import SeverityResult
from .outcome import SimulatedOutcome
from .priority import SEVERITY_PRIORITY_MAP, PRIORITY_ACTION_MAP


@dataclass(frozen=True)
class Decision:
    """A recommended action with full explainability.

    Immutable value object representing a decision recommendation.
    Contains all information needed to understand, audit, and act.

    Attributes:
        action: Recommended action code ("monitor", "investigate", "intervene", "escalate")
        priority: Priority level 1-4 (1=highest/critical, 4=lowest/info)
        confidence: Confidence in this decision [0, 1]
        reason: Human-readable justification
        strategy_used: Which strategy produced this decision
        simulated_outcomes: Evidence from scenario simulation
        source_ml_outputs: References to upstream ML results
        audit_trace_id: Trace ID for audit logging
        metadata: Additional decision metadata
    """

    action: str = "monitor"
    priority: int = 4
    confidence: float = 0.0
    reason: str = ""
    strategy_used: str = "unknown"
    simulated_outcomes: List[SimulatedOutcome] = field(default_factory=list)
    source_ml_outputs: Dict[str, Any] = field(default_factory=dict)
    audit_trace_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Priority constants for validation
    PRIORITY_CRITICAL = 1
    PRIORITY_HIGH = 2
    PRIORITY_MEDIUM = 3
    PRIORITY_LOW = 4

    @property
    def is_actionable(self) -> bool:
        """True if this decision requires action (priority <= 2)."""
        return self.priority <= self.PRIORITY_HIGH

    @property
    def has_simulated_outcomes(self) -> bool:
        """True if simulated outcomes are available."""
        return len(self.simulated_outcomes) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for API responses."""
        return {
            "action": self.action,
            "priority": self.priority,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "strategy_used": self.strategy_used,
            "simulated_outcomes": [o.to_dict() for o in self.simulated_outcomes],
            "source_ml_outputs": self.source_ml_outputs,
            "audit_trace_id": self.audit_trace_id,
            "is_actionable": self.is_actionable,
            "metadata": dict(self.metadata) if self.metadata else {},
        }

    @classmethod
    def noop(cls, series_id: str = "unknown", reason: str = "No decision needed") -> "Decision":
        """Factory for a no-op decision (monitor priority)."""
        return cls(
            action="monitor",
            priority=cls.PRIORITY_LOW,
            confidence=1.0,
            reason=reason,
            strategy_used="noop",
            source_ml_outputs={"series_id": series_id},
        )

    @classmethod
    def from_severity(
        cls,
        severity: SeverityResult,
        series_id: str = "unknown",
        strategy: str = "passthrough",
    ) -> "Decision":
        """Factory that derives decision directly from severity result.

        MVP passthrough: maps severity directly to decision without strategy logic.
        """
        priority = SEVERITY_PRIORITY_MAP.get(severity.severity, cls.PRIORITY_LOW)
        action = PRIORITY_ACTION_MAP.get(priority, "monitor")

        return cls(
            action=action,
            priority=priority,
            confidence=0.8 if severity.action_required else 0.95,
            reason=severity.recommended_action,
            strategy_used=strategy,
            source_ml_outputs={"series_id": series_id, "severity": severity.severity},
        )
