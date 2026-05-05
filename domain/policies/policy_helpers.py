"""Helper functions for ThresholdPolicy."""
from __future__ import annotations
from typing import Optional
from ..entities.results.anomaly import AnomalySeverity


def _severity_to_risk_level(severity: AnomalySeverity) -> str:
    mapping = {
        AnomalySeverity.NONE: "NONE",
        AnomalySeverity.LOW: "LOW",
        AnomalySeverity.MEDIUM: "MEDIUM",
        AnomalySeverity.HIGH: "HIGH",
        AnomalySeverity.CRITICAL: "CRITICAL",
    }
    return mapping[severity]


def _regime_policy(policy, regime: Optional[str]):
    """Return regime-specific policy override if configured."""
    if regime and policy.regime_overrides and regime in policy.regime_overrides:
        return policy.regime_overrides[regime]
    return policy
