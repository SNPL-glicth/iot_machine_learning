"""Text severity classification for ThresholdPolicy."""
from __future__ import annotations
from typing import Optional
from ..entities.results.anomaly import AnomalySeverity
from .action_builders import build_text_action
from .policy_helpers import _severity_to_risk_level
from .policy_result import SeverityPolicyResult


def classify_text(
    policy,
    *,
    urgency_score: float,
    sentiment_weight: float,
    impact_score: float,
    domain: str = "general",
    n_categories_hit: int = 0,
    urgency_override: bool = False,
) -> SeverityPolicyResult:
    """3-axis composite severity for text/document analysis.

    Replaces severity_mapper.classify_text_severity() logic.
    """
    w_u, w_s, w_i = policy.text_weights
    composite = w_u * urgency_score + w_s * sentiment_weight + w_i * impact_score

    info_max, warning_min, critical_min = policy.text_thresholds
    if n_categories_hit >= 3:
        composite = max(composite, critical_min)
    if urgency_override:
        composite = max(composite, warning_min)

    severity = policy.classify_score(composite)
    severity_label = policy.classify_score_label(composite)

    if composite < info_max:
        severity_label = "info"
        severity = AnomalySeverity.NONE
    elif composite < warning_min:
        severity_label = "info"
        severity = AnomalySeverity.LOW
    elif composite < critical_min:
        severity_label = "warning"
        severity = AnomalySeverity.MEDIUM
    else:
        severity_label = "critical"
        severity = AnomalySeverity.CRITICAL

    action_required = severity_label != "info"
    recommended_action = build_text_action(severity_label, domain)

    return SeverityPolicyResult(
        severity=severity,
        severity_label=severity_label,
        risk_level=_severity_to_risk_level(severity),
        action_required=action_required,
        recommended_action=recommended_action,
        composite_score=round(composite, 3),
    )
