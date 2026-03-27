"""DecisionContext — aggregated ML outputs for decision-making.

Domain-pure dataclass capturing all relevant information from
upstream ML pipeline without dependencies on specific ML types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ....domain.services.severity_rules import SeverityResult
from .outcome import SimulatedOutcome


@dataclass(frozen=True)
class DecisionContext:
    """Aggregated ML outputs required for decision-making.

    Captures all relevant information from upstream ML pipeline
    without dependencies on specific ML types.

    Attributes:
        series_id: Identifier for the entity being analyzed
        severity: Severity classification from severity_rules
        confidence: Overall ML confidence [0, 1]
        is_anomaly: Whether an anomaly was detected
        anomaly_score: Anomaly score [0, 1] if applicable
        patterns: List of detected patterns with metadata
        predicted_value: Predicted future value if available
        trend: Detected trend direction
        monte_carlo_outcomes: Optional pre-computed scenario outcomes
        domain: Detected/assigned domain (infrastructure, security, etc.)
        audit_trace_id: Trace ID for audit logging
        extra: Additional ML outputs not captured above
    """

    series_id: str
    severity: SeverityResult = field(
        default_factory=lambda: SeverityResult(
            risk_level="NONE",
            severity="info",
            action_required=False,
            recommended_action="No action required",
        )
    )
    confidence: float = 0.0
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    patterns: List[Dict[str, Any]] = field(default_factory=list)
    predicted_value: Optional[float] = None
    trend: str = "stable"
    monte_carlo_outcomes: Optional[List[SimulatedOutcome]] = None
    domain: str = ""
    audit_trace_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_monte_carlo(self) -> bool:
        """True if Monte Carlo outcomes are available."""
        return self.monte_carlo_outcomes is not None and len(self.monte_carlo_outcomes) > 0

    @property
    def has_critical_pattern(self) -> bool:
        """True if any pattern has critical severity hint."""
        return any(
            p.get("severity_hint", "").lower() == "critical"
            for p in self.patterns
        )

    @property
    def action_required(self) -> bool:
        """True if upstream severity indicates action is required."""
        return self.severity.action_required

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "series_id": self.series_id,
            "severity": {
                "risk_level": self.severity.risk_level,
                "severity": self.severity.severity,
                "action_required": self.severity.action_required,
                "recommended_action": self.severity.recommended_action,
            },
            "confidence": round(self.confidence, 4),
            "is_anomaly": self.is_anomaly,
            "anomaly_score": round(self.anomaly_score, 4),
            "patterns": self.patterns,
            "predicted_value": self.predicted_value,
            "trend": self.trend,
            "monte_carlo_outcomes": (
                [o.to_dict() for o in self.monte_carlo_outcomes]
                if self.monte_carlo_outcomes else None
            ),
            "domain": self.domain,
            "audit_trace_id": self.audit_trace_id,
        }
