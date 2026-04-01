"""TextDecisionOutput — clear and concise text analysis output.

Eliminates the caller's burden of inferring the decision from 10+ optional fields.
Single source of truth for text analysis results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass(frozen=True)
class TextDecisionOutput:
    """Clear, concise text analysis output with mandatory fields.

    Attributes:
        decision: High-level decision category
            - "normal": Text contains no concerning patterns
            - "anomaly": Anomaly detected in text
            - "critical": Critical severity detected
            - "info": Informational only
        confidence: Overall confidence score [0, 1]
        verdict: Human-readable single-sentence summary of the analysis
        severity: Impact severity
            - "critical": Immediate attention required
            - "warning": Attention may be required
            - "info": Informational
            - "unknown": Unable to determine
        domain: Auto-detected document domain (from TextCognitiveEngine)
        metadata: Optional dict with all auxiliary data
    """

    decision: Literal["normal", "anomaly", "critical", "info"]
    confidence: float
    verdict: str
    severity: Literal["critical", "warning", "info", "unknown"]
    domain: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict — always includes all 5 mandatory fields + metadata.

        Returns dict with:
        - decision, confidence, verdict, severity, domain (mandatory)
        - metadata (dict block with everything else)
        """
        return {
            "decision": self.decision,
            "confidence": round(self.confidence, 4),
            "verdict": self.verdict,
            "severity": self.severity,
            "domain": self.domain,
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
    def from_text_result(
        cls,
        result_dict: Dict[str, Any],
        default_confidence: float = 0.0,
    ) -> "TextDecisionOutput":
        """Factory method to construct TextDecisionOutput from TextCognitiveResult.to_dict().

        Extracts values from:
        - severity.risk_level / severity.severity
        - conclusion: provides verdict
        - confidence: overall confidence
        - domain: detected domain

        Args:
            result_dict: Dict from TextCognitiveResult.to_dict()
            default_confidence: Default if confidence not found

        Returns:
            TextDecisionOutput with extracted or defaulted values
        """
        # Extract severity info
        severity_obj = result_dict.get("severity", {})
        risk_level = severity_obj.get("risk_level", "LOW").lower()
        severity_level = severity_obj.get("severity", "INFO").lower()

        # Normalize severity to allowed values
        severity_map = {
            "critical": "critical",
            "high": "critical",
            "severe": "critical",
            "warning": "warning",
            "medium": "warning",
            "moderate": "warning",
            "info": "info",
            "low": "info",
            "normal": "info",
            "none": "info",
            "unknown": "unknown",
        }
        severity = severity_map.get(severity_level, "unknown")
        if severity == "unknown":
            severity = severity_map.get(risk_level, "unknown")

        # Determine decision based on severity and risk
        decision: Literal["normal", "anomaly", "critical", "info"] = "info"
        if risk_level == "high" or severity == "critical":
            decision = "critical"
        elif risk_level == "medium" or severity == "warning":
            decision = "anomaly"
        elif severity == "info":
            decision = "normal" if risk_level == "low" else "info"

        # Extract verdict from conclusion
        verdict = result_dict.get("conclusion", "")
        if not verdict:
            verdict = f"text analysis: {severity} severity, {risk_level} risk"

        # Extract confidence
        confidence = result_dict.get("confidence", default_confidence)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = default_confidence

        # Extract domain
        domain = result_dict.get("domain", "general")

        # Build metadata from everything else
        metadata: Dict[str, Any] = {}
        for key, value in result_dict.items():
            if key not in ("conclusion", "confidence", "domain", "severity"):
                metadata[key] = value

        return cls(
            decision=decision,
            confidence=confidence,
            verdict=verdict,
            severity=severity,
            domain=domain,
            metadata=metadata,
        )
