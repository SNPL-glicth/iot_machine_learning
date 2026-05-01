"""Domain-specific action catalog for intelligent recommendations."""

from __future__ import annotations

from typing import Dict, List


DOMAIN_ACTIONS: Dict[str, Dict[str, List[str]]] = {
    "infrastructure": {
        "critical": [
            "→ Restart affected node immediately", "→ Check sensor readings and thresholds",
            "→ Reduce system load to prevent cascade failure", "→ Notify on-call engineer",
            "→ Activate incident response protocol",
        ],
        "warning": [
            "→ Schedule infrastructure review within 24 hours", "→ Monitor affected components for further degradation",
            "→ Verify backup systems are operational", "→ Document current state for trend analysis",
        ],
        "info": [
            "→ Continue standard monitoring", "→ Log for capacity planning review",
            "→ No immediate infrastructure action required",
        ],
    },
    "security": {
        "critical": [
            "→ Isolate affected systems immediately", "→ Revoke all active sessions",
            "→ Notify security team and authorities", "→ Begin forensic analysis",
            "→ Activate breach response protocol",
        ],
        "warning": [
            "→ Investigate potential security concern within 12 hours", "→ Review access logs and verify system integrity",
            "→ Enable enhanced monitoring on affected assets", "→ Document findings for security audit trail",
        ],
        "info": [
            "→ Log for security audit trail", "→ Continue standard security monitoring",
            "→ No immediate security action needed",
        ],
    },
    "trading": {
        "critical": [
            "→ Close position immediately", "→ Reduce exposure by 60%",
            "→ Activate stop-loss on all instruments with delta > 0.7", "→ Suspend algorithmic operations",
            "→ Notify risk committee — Level 3 protocol",
        ],
        "warning": [
            "→ Review position sizing and risk limits", "→ Monitor volatility indicators closely",
            "→ Prepare hedging strategies", "→ Document trading anomalies for compliance",
        ],
        "info": [
            "→ Continue standard trading operations", "→ Log for trading desk review",
            "→ No immediate trading action required",
        ],
    },
    "operations": {
        "critical": [
            "→ Stop production line immediately", "→ Escalate to maintenance team",
            "→ Activate emergency maintenance protocol", "→ Document failure for root cause analysis",
        ],
        "warning": [
            "→ Review operational procedure before proceeding", "→ Validate preconditions and ensure rollback plan exists",
            "→ Schedule preventive maintenance within 48 hours", "→ Monitor operational KPIs for degradation",
        ],
        "info": [
            "→ Proceed with standard operational procedures", "→ Log for operations review",
            "→ No immediate operations action required",
        ],
    },
    "business": {
        "critical": [
            "→ Notify executive board immediately", "→ Activate crisis management protocol",
            "→ Prepare stakeholder communication", "→ Assess legal and regulatory exposure",
        ],
        "warning": [
            "→ Schedule review with relevant stakeholders within 48 hours", "→ Document impact assessment",
            "→ Prepare contingency plans", "→ Monitor contractual obligations",
        ],
        "info": [
            "→ No immediate business action required", "→ Continue standard business operations",
            "→ Log for quarterly business review",
        ],
    },
    "general": {
        "critical": ["→ Immediate intervention required"],
        "warning": ["→ Schedule review within 24 hours"],
        "info": ["→ Continue standard monitoring"],
    },
}


def get_actions_for_domain(domain: str, severity: str, max_actions: int = 3) -> List[str]:
    """Get domain-specific actions for a given severity level."""
    domain_lower = domain.lower() if domain else "general"
    severity_lower = severity.lower() if severity else "info"
    domain_data = DOMAIN_ACTIONS.get(domain_lower, DOMAIN_ACTIONS["general"])
    actions = domain_data.get(severity_lower, domain_data.get("info", []))
    return actions[:max_actions]
