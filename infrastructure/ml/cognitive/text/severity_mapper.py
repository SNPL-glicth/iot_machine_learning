"""TextSeverityMapper — maps text urgency/sentiment to SeverityResult.

Produces the same ``SeverityResult`` dataclass used by the IoT pipeline
(``domain.services.severity_rules``) but without requiring the
IoT-specific ``Threshold`` entity.

No imports from ml_service — only domain layer.
Single entry point: ``classify_text_severity()``.
"""

from __future__ import annotations

from iot_machine_learning.domain.services.severity_rules import SeverityResult


def classify_text_severity(
    *,
    urgency_score: float,
    urgency_severity: str,
    sentiment_label: str,
    has_critical_keywords: bool = False,
    domain: str = "general",
) -> SeverityResult:
    """Classify text document severity from analysis scores.

    Args:
        urgency_score: Urgency level [0, 1].
        urgency_severity: Pre-computed urgency severity
            (``"critical"``, ``"warning"``, ``"info"``).
        sentiment_label: Sentiment classification
            (``"positive"``, ``"negative"``, ``"neutral"``).
        has_critical_keywords: Whether critical urgency keywords
            were detected in the text.
        domain: Classified document domain for action text.

    Returns:
        ``SeverityResult`` with risk_level, severity,
        action_required, and recommended_action.
    """
    severity, risk_level = _compute_severity(
        urgency_score, urgency_severity, sentiment_label, has_critical_keywords
    )

    action_required = severity != "info"
    recommended_action = _build_action(severity, domain)

    return SeverityResult(
        risk_level=risk_level,
        severity=severity,
        action_required=action_required,
        recommended_action=recommended_action,
    )


def _compute_severity(
    urgency_score: float,
    urgency_severity: str,
    sentiment_label: str,
    has_critical_keywords: bool,
) -> tuple:
    """Compute severity and risk level from text signals.

    Returns:
        Tuple of (severity, risk_level).
    """
    # Critical: explicit critical urgency OR critical keywords + negative sentiment
    if urgency_severity == "critical":
        return "critical", "HIGH"
    if has_critical_keywords and sentiment_label == "negative":
        return "critical", "HIGH"

    # Warning: moderate urgency OR negative sentiment with elevated urgency
    if urgency_severity == "warning":
        return "warning", "MEDIUM"
    if sentiment_label == "negative" and urgency_score > 0.3:
        return "warning", "MEDIUM"

    # Info
    if urgency_score > 0.2 or sentiment_label == "negative":
        return "info", "LOW"

    return "info", "NONE"


def _build_action(severity: str, domain: str) -> str:
    """Build human-readable recommended action."""
    domain_label = domain if domain != "general" else "document"

    if severity == "critical":
        return (
            f"Critical issues detected in {domain_label}. "
            "Immediate review and action required."
        )
    if severity == "warning":
        return (
            f"Concerns identified in {domain_label}. "
            "Schedule review and monitor closely."
        )
    return "No immediate action required. Continue standard monitoring."
