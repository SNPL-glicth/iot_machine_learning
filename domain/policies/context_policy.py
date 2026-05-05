"""Contextual severity classification for ThresholdPolicy."""
from __future__ import annotations
from typing import Optional
from ..entities.results.anomaly import AnomalySeverity
from ..entities.severity import SeverityResult
from .action_builders import build_action
from .policy_helpers import _regime_policy, _severity_to_risk_level
from .policy_result import SeverityPolicyResult


def classify_with_context(
    policy,
    *,
    score: float,
    is_anomaly: bool = False,
    risk_level: str = "NONE",
    out_of_physical_range: bool = False,
    regime: Optional[str] = None,
    label: str = "",
) -> SeverityPolicyResult:
    """Full severity classification with all context.

    Replaces:
    - severity_rules.compute_severity()
    - severity_rules.classify_severity_agnostic()
    - adaptive_thresholds classify_severity (partial)
    """
    effective = _regime_policy(policy, regime)
    severity = effective.classify_score(score)
    severity_label = effective.classify_score_label(score)

    if out_of_physical_range:
        severity = AnomalySeverity.CRITICAL
        severity_label = "critical"

    rl = (risk_level or "").upper()
    if is_anomaly and rl == "HIGH":
        severity = AnomalySeverity.CRITICAL
        severity_label = "critical"

    if is_anomaly or rl == "HIGH":
        if severity.value not in {"critical", "high"}:
            severity = AnomalySeverity.HIGH
            severity_label = "warning"

    action_required = severity_label != "info"
    recommended_action = build_action(
        severity_label=severity_label,
        risk_level=_severity_to_risk_level(severity),
        label=label,
    )

    return SeverityPolicyResult(
        severity=severity,
        severity_label=severity_label,
        risk_level=_severity_to_risk_level(severity),
        action_required=action_required,
        recommended_action=recommended_action,
        regime=regime,
    )
