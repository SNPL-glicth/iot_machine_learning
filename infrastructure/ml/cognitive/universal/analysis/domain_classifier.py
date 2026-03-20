"""Domain classification from content analysis."""

from __future__ import annotations

from typing import Any, Dict, List

from .types import InputType


DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "infrastructure": [
        "server", "cpu", "memory", "disk", "network", "node",
        "cluster", "deploy", "container", "kubernetes", "latency",
        "bandwidth", "throughput", "load", "capacity", "storage",
    ],
    "security": [
        "vulnerability", "breach", "unauthorized", "firewall",
        "intrusion", "malware", "exploit", "authentication",
        "encryption", "certificate", "attack", "threat", "access",
    ],
    "operations": [
        "incident", "outage", "downtime", "maintenance",
        "escalation", "sla", "recovery", "alert", "ticket",
        "oncall", "runbook", "postmortem", "deploy",
    ],
    "business": [
        "revenue", "cost", "budget", "forecast", "margin",
        "growth", "kpi", "target", "profit", "contract",
        "customer", "sales", "market", "invoice",
    ],
    "trading": [
        "price", "volume", "volatility", "bid", "ask", "spread",
        "order", "execution", "liquidity", "risk", "position",
        "hedge", "derivative", "futures", "options", "market",
    ],
}


def classify_domain(
    raw_data: Any,
    input_type: InputType,
    metadata: Dict[str, Any],
    hint: str = "",
) -> str:
    """Classify document/data domain.

    Args:
        raw_data: Original input
        input_type: Detected InputType
        metadata: From input_detector
        hint: Optional domain override

    Returns:
        Domain string (infrastructure, security, trading, operations, business, general)
    """
    if hint and hint in DOMAIN_KEYWORDS:
        return hint
    
    if input_type == InputType.TEXT:
        return _classify_text_domain(str(raw_data))
    
    if input_type == InputType.NUMERIC:
        return _classify_numeric_domain(raw_data, metadata)
    
    if input_type == InputType.TABULAR:
        return _classify_tabular_domain(raw_data, metadata)
    
    if input_type == InputType.MIXED:
        if isinstance(raw_data, str):
            return _classify_text_domain(raw_data)
    
    return "general"


def _classify_text_domain(text: str) -> str:
    """Keyword-based domain classification for text."""
    text_lower = text.lower()
    scores: Dict[str, int] = {}
    
    for domain_name, keywords in DOMAIN_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > 0:
            scores[domain_name] = count
    
    if not scores:
        return "general"
    
    return max(scores, key=scores.get)  # type: ignore[arg-type]


def _classify_numeric_domain(values: list, metadata: Dict[str, Any]) -> str:
    """Pattern-based domain classification for numeric data.
    
    Heuristics:
        - High variance (CV > 0.5) → trading
        - Low variance (CV < 0.1) → operations (stable metrics)
        - Otherwise → general
    """
    if not values or len(values) < 3:
        return "general"
    
    try:
        mean = sum(values) / len(values)
        if abs(mean) < 1e-9:
            return "general"
        
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance ** 0.5
        cv = std / abs(mean)
        
        if cv > 0.5:
            return "trading"
        if cv < 0.1:
            return "operations"
    except Exception:
        pass
    
    return "general"


def _classify_tabular_domain(data: dict, metadata: Dict[str, Any]) -> str:
    """Column-name-based domain classification."""
    column_names = metadata.get("column_names", [])
    
    if not column_names:
        return "general"
    
    col_text = " ".join(str(c).lower() for c in column_names)
    
    scores: Dict[str, int] = {}
    for domain_name, keywords in DOMAIN_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in col_text)
        if count > 0:
            scores[domain_name] = count
    
    if not scores:
        return "general"
    
    return max(scores, key=scores.get)  # type: ignore[arg-type]
