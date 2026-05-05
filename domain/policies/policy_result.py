"""Value object for threshold policy classification results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from ..entities.results.anomaly import AnomalySeverity


@dataclass(frozen=True)
class SeverityPolicyResult:
    """Output of ThresholdPolicy.classify_with_context().

    Combines all severity dimensions into a single value object.
    """

    severity: AnomalySeverity
    severity_label: str  # "info" | "warning" | "critical"
    risk_level: str  # "NONE" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    action_required: bool
    recommended_action: str = ""
    composite_score: Optional[float] = None  # For text/3-axis classification
    regime: Optional[str] = None  # Optional regime context

    def to_dict(self) -> Dict:
        return {
            "severity": self.severity.value,
            "severity_label": self.severity_label,
            "risk_level": self.risk_level,
            "action_required": self.action_required,
            "recommended_action": self.recommended_action,
            "composite_score": self.composite_score,
            "regime": self.regime,
        }
