"""DecisionOutput — clear and concise prediction output.

Eliminates the caller's burden of inferring the decision from 10+ optional fields.
Single source of truth for what happened, how confident we are, and what to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass(frozen=True)
class DecisionOutput:
    """Clear, concise prediction output with mandatory fields.

    Attributes:
        decision: High-level decision category
            - "normal": System operating normally
            - "anomaly": Anomaly detected
            - "out_of_domain": Data outside domain boundary
            - "degraded": System degraded (low confidence/suppressed action)
        confidence: Calibrated confidence score [0, 1] (from EJE 6)
        verdict: Human-readable single-sentence summary
            (from UnifiedNarrative.primary_verdict if available)
        severity: Impact severity
            - "critical": Immediate attention required
            - "warning": Attention may be required
            - "info": Informational
            - "unknown": Unable to determine
        action_required: Whether action is needed
            (from ActionGuard.action_allowed)
        action: Recommended action string, or None if no action
            (from ActionGuard.final_action)
        metadata: Optional dict with all auxiliary data
            (engine_decision, coherence_check, calibration_report,
            boundary_check, action_guard, unified_narrative, etc.)
    """

    decision: Literal["normal", "anomaly", "out_of_domain", "degraded"]
    confidence: float
    verdict: str
    severity: Literal["critical", "warning", "info", "unknown"]
    action_required: bool
    action: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict — always includes all 6 mandatory fields + metadata.

        Returns dict with:
        - decision, confidence, verdict, severity (mandatory)
        - action_required, action (mandatory)
        - metadata (dict block with everything else)
        """
        return {
            "decision": self.decision,
            "confidence": round(self.confidence, 4),
            "verdict": self.verdict,
            "severity": self.severity,
            "action_required": self.action_required,
            "action": self.action,
            "metadata": self.metadata,
        }

    def to_summary(self) -> Dict[str, Any]:
        """Minimal summary — exactly 3 fields for quick consumption.

        Returns dict with:
        - decision: str
        - confidence: float
        - verdict: str
        """
        return {
            "decision": self.decision,
            "confidence": round(self.confidence, 4),
            "verdict": self.verdict,
        }

    @classmethod
    def from_metadata(
        cls,
        metadata: Dict[str, Any],
        series_id: str = "unknown",
        predicted_value: Optional[float] = None,
        default_confidence: float = 0.0,
    ) -> "DecisionOutput":
        """Factory method to construct DecisionOutput from prediction metadata.

        Extracts values from:
        - boundary_check: determines out_of_domain
        - unified_narrative: provides verdict, severity
        - action_guard: provides action_required, action
        - calibration_report: provides confidence

        Args:
            metadata: Full metadata dict from pipeline
            series_id: Series identifier for default verdict
            predicted_value: Predicted value for default verdict
            default_confidence: Default confidence if not found

        Returns:
            DecisionOutput with extracted or defaulted values
        """
        # Check for out of domain
        boundary_check = metadata.get("boundary_check", {})
        if boundary_check.get("within_domain") is False:
            return cls(
                decision="out_of_domain",
                confidence=0.0,
                verdict=boundary_check.get(
                    "rejection_reason", "data out of domain"
                ),
                severity="unknown",
                action_required=False,
                action=None,
                metadata=metadata,
            )

        # Extract from unified_narrative if available
        unified_narrative = metadata.get("unified_narrative", {})
        verdict = unified_narrative.get("primary_verdict")
        severity = unified_narrative.get("severity", "unknown").lower()

        # Normalize severity to allowed values
        if severity not in ("critical", "warning", "info", "unknown"):
            severity_map = {
                "critical": "critical",
                "alert": "critical",
                "error": "critical",
                "warning": "warning",
                "warn": "warning",
                "info": "info",
                "normal": "info",
                "ok": "info",
            }
            severity = severity_map.get(severity, "unknown")

        # Default verdict if not from unified_narrative
        if not verdict:
            if predicted_value is not None:
                verdict = f"predicted value {predicted_value:.2f} for series {series_id}"
            else:
                verdict = f"prediction for series {series_id}"

        # Extract confidence from calibration or use default
        calibration_report = metadata.get("calibration_report", {})
        confidence = calibration_report.get("calibrated", default_confidence)
        if confidence is None or confidence <= 0:
            confidence = default_confidence

        # Determine decision based on severity
        decision: Literal["normal", "anomaly", "out_of_domain", "degraded"] = "normal"
        if severity == "critical":
            decision = "anomaly"
        elif severity == "warning":
            decision = "anomaly"

        # Check if degraded (action suppressed or low confidence)
        action_guard = metadata.get("action_guard", {})
        if action_guard.get("action_allowed") is False:
            decision = "degraded"

        # Extract action info
        action_required = action_guard.get("action_required", False)
        if action_guard.get("action_allowed") is False:
            action_required = False

        action = action_guard.get("final_action")
        if not action:
            action = None

        return cls(
            decision=decision,
            confidence=confidence,
            verdict=verdict,
            severity=severity,
            action_required=action_required,
            action=action,
            metadata=metadata,
        )
