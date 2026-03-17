"""Centralized keyword configuration for text analysis.

All keyword lists live here.  Logic modules import from this file
instead of defining their own constants.  To add/remove keywords,
edit ONLY this file.
"""

from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# Urgency keywords (Spanish)
# ---------------------------------------------------------------------------

URGENCY_KEYWORDS_ES: List[str] = [
    "error", "falla", "crítico", "alerta", "urgente", "caída",
    "pérdida", "crisis", "incidente", "interrupción", "degradación",
    "timeout", "excepción", "fatal", "pánico", "sobrecarga",
]

# ---------------------------------------------------------------------------
# Urgency keywords (English)
# ---------------------------------------------------------------------------

URGENCY_KEYWORDS_EN: List[str] = [
    "error", "failure", "critical", "alert", "urgent", "down",
    "loss", "crisis", "incident", "outage", "degradation",
    "timeout", "exception", "fatal", "panic", "overload",
]

# ---------------------------------------------------------------------------
# Sentiment — positive words (ES + EN)
# ---------------------------------------------------------------------------

POSITIVE_WORDS: List[str] = [
    "bueno", "excelente", "éxito", "mejora", "bien", "estable",
    "óptimo", "resuelto", "correcto", "recuperado",
    "good", "excellent", "success", "improved", "stable",
    "optimal", "resolved", "correct", "recovered",
]

# ---------------------------------------------------------------------------
# Sentiment — negative words (ES + EN)
# ---------------------------------------------------------------------------

NEGATIVE_WORDS: List[str] = [
    "malo", "error", "problema", "falla", "inestable", "degradado",
    "lento", "fallido", "rechazado", "perdido",
    "bad", "problem", "failure", "unstable", "degraded",
    "slow", "failed", "rejected", "lost",
]

# ---------------------------------------------------------------------------
# Domain classification keywords
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "infrastructure": [
        "server", "servidor", "cpu", "memory", "memoria", "disk", "disco",
        "network", "red", "node", "nodo", "cluster", "database", "base de datos",
        "cooling", "refrigeración", "temperature", "temperatura", "rack",
        "power", "energía", "uptime", "latency", "latencia", "bandwidth",
    ],
    "security": [
        "breach", "brecha", "vulnerability", "vulnerabilidad", "attack",
        "ataque", "unauthorized", "no autorizado", "malware", "phishing",
        "encryption", "cifrado", "firewall", "intrusion", "intrusión",
        "credential", "credencial", "exploit", "ransomware",
    ],
    "operations": [
        "deploy", "despliegue", "release", "versión", "rollback",
        "maintenance", "mantenimiento", "update", "actualización",
        "migration", "migración", "backup", "respaldo", "restore",
        "schedule", "cron", "pipeline", "ci/cd",
    ],
    "business": [
        "revenue", "ingreso", "cost", "costo", "budget", "presupuesto",
        "client", "cliente", "contract", "contrato", "sla", "compliance",
        "cumplimiento", "audit", "auditoría", "regulation", "regulación",
    ],
}

# ---------------------------------------------------------------------------
# Severity → recommended actions mapping
# ---------------------------------------------------------------------------

ACTION_TEMPLATES: Dict[str, Dict[str, str]] = {
    "infrastructure": {
        "critical": (
            "Immediate review of affected infrastructure components required. "
            "Escalate to infrastructure team. Verify monitoring dashboards "
            "and prepare failover if available."
        ),
        "warning": (
            "Schedule infrastructure review within 24 hours. "
            "Monitor affected components for further degradation."
        ),
        "info": "No immediate infrastructure action required.",
    },
    "security": {
        "critical": (
            "Activate incident response protocol immediately. "
            "Isolate affected systems, preserve evidence, notify security team. "
            "Assess blast radius and credential exposure."
        ),
        "warning": (
            "Investigate potential security concern within 12 hours. "
            "Review access logs and verify system integrity."
        ),
        "info": "Log for security audit trail. No immediate action needed.",
    },
    "operations": {
        "critical": (
            "Halt current deployment/operation if in progress. "
            "Assess rollback requirements. Notify operations team lead."
        ),
        "warning": (
            "Review operational procedure before proceeding. "
            "Validate preconditions and ensure rollback plan exists."
        ),
        "info": "Proceed with standard operational procedures.",
    },
    "business": {
        "critical": (
            "Escalate to management immediately. "
            "Assess financial and contractual impact. "
            "Prepare stakeholder communication."
        ),
        "warning": (
            "Schedule review with relevant stakeholders within 48 hours. "
            "Document impact assessment."
        ),
        "info": "No immediate business action required.",
    },
    "general": {
        "critical": (
            "Immediate attention required. Review document contents "
            "and escalate to the appropriate team."
        ),
        "warning": (
            "Review within 24-48 hours and assess if further action is needed."
        ),
        "info": "Informational document. No action required.",
    },
}
