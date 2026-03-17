"""Urgency detection for text documents.

Pure function — no side effects, no I/O.
Maps urgency score to formal severity via ``classify_severity_agnostic``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .keyword_config import URGENCY_KEYWORDS_ES, URGENCY_KEYWORDS_EN

# Lazy import for severity classification (may not be available)
_severity_available = True
try:
    from .....domain.services.severity_rules import classify_severity_agnostic
except Exception:
    _severity_available = False


@dataclass(frozen=True)
class UrgencyResult:
    """Result of urgency analysis.

    Attributes:
        score: Urgency score in [0.0, 1.0].
        total_hits: Total keyword occurrence count.
        hits: Top keyword hits with counts.
        severity: Formal severity level (``"info"`` | ``"warning"`` | ``"critical"``).
        action_required: Whether the severity warrants action.
    """

    score: float
    total_hits: int
    hits: List[Dict[str, Any]]
    severity: str = "info"
    action_required: bool = False


def compute_urgency(text: str) -> UrgencyResult:
    """Compute urgency from keyword matching with severity mapping.

    Args:
        text: Raw text content.

    Returns:
        ``UrgencyResult`` with score, hits, and formal severity.
    """
    text_lower = text.lower()

    all_keywords = URGENCY_KEYWORDS_ES + URGENCY_KEYWORDS_EN
    # Deduplicate (e.g. "error" appears in both lists)
    seen = set()
    unique_keywords = []
    for kw in all_keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)

    hits: List[Dict[str, Any]] = []
    for kw in unique_keywords:
        count = text_lower.count(kw)
        if count > 0:
            hits.append({"keyword": kw, "count": count})

    total_hits = sum(h["count"] for h in hits)
    score = min(1.0, total_hits / 10.0)

    # Map to formal severity
    severity = "info"
    action_required = False

    if _severity_available:
        try:
            result = classify_severity_agnostic(
                value=score,
                anomaly=score > 0.7,
                label="urgencia de texto",
            )
            severity = result.severity
            action_required = result.action_required
        except Exception:
            # Fallback to simple thresholds
            severity, action_required = _fallback_severity(score)
    else:
        severity, action_required = _fallback_severity(score)

    return UrgencyResult(
        score=round(score, 3),
        total_hits=total_hits,
        hits=hits[:10],
        severity=severity,
        action_required=action_required,
    )


def _fallback_severity(score: float) -> tuple:
    """Simple threshold-based severity when domain service unavailable."""
    if score > 0.7:
        return "critical", True
    if score > 0.4:
        return "warning", True
    return "info", False
