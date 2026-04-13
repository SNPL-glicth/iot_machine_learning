"""TextSeverityMapper — 3-axis severity classification.

Combines three independent axes to compute real-world severity:

    **urgency**   × 0.30  — keyword-based urgency score [0, 1]
    **sentiment** × 0.20  — negative sentiment weight [0, 1]
    **impact**    × 0.50  — hard impact signals (SLA breach, extreme
                            metrics, temporal risk, cascade failure)

Produces the same ``SeverityResult`` dataclass used by the IoT pipeline
(``domain.services.severity_rules``) but without requiring the
IoT-specific ``Threshold`` entity.

No imports from ml_service — only domain layer + sibling modules.
Single entry point: ``classify_text_severity()``.
"""

from __future__ import annotations

from typing import Optional

from iot_machine_learning.domain.services.severity_rules import SeverityResult

from .impact_detector import ImpactSignalResult, detect_impact_signals

# Axis weights for the composite severity score
_W_URGENCY = 0.45      # INCREASED: Urgency should have significant weight
_W_SENTIMENT = 0.20
_W_IMPACT = 0.35       # REDUCED: Balance with higher urgency weight

# Severity thresholds on composite score [0, 1]
_THRESHOLD_CRITICAL = 0.55   # REDUCED: Make critical achievable with high urgency
_THRESHOLD_WARNING = 0.35    # REDUCED: Balance with higher urgency weight


def classify_text_severity(
    *,
    urgency_score: float,
    urgency_severity: str,
    sentiment_label: str,
    has_critical_keywords: bool = False,
    domain: str = "general",
    full_text: str = "",
    impact_result: Optional[ImpactSignalResult] = None,
) -> SeverityResult:
    """Classify text document severity from analysis scores + impact signals.

    Uses a 3-axis formula::

        composite = urgency * 0.30 + sentiment * 0.20 + impact * 0.50

    Args:
        urgency_score: Urgency level [0, 1].
        urgency_severity: Pre-computed urgency severity
            (``"critical"``, ``"warning"``, ``"info"``).
        sentiment_label: Sentiment classification
            (``"positive"``, ``"negative"``, ``"neutral"``).
        has_critical_keywords: Whether critical urgency keywords
            were detected in the text.
        domain: Classified document domain for action text.
        full_text: Full document text for impact signal scanning.
            If empty and ``impact_result`` is ``None``, impact axis
            scores zero (backward-compatible).
        impact_result: Pre-computed ``ImpactSignalResult``. If
            provided, ``full_text`` is not re-scanned.

    Returns:
        ``SeverityResult`` with risk_level, severity,
        action_required, and recommended_action.
    """
    # ── Impact axis ──
    if impact_result is None and full_text:
        impact_result = detect_impact_signals(full_text)

    impact_score = impact_result.score if impact_result is not None else 0.0

    # ── Sentiment axis ──
    sentiment_weight = _sentiment_to_weight(sentiment_label)

    # ── Composite score ──
    composite = (
        _W_URGENCY * urgency_score
        + _W_SENTIMENT * sentiment_weight
        + _W_IMPACT * impact_score
    )

    # ── Hard overrides (urgency-based) ──
    # Impact override: 3+ hard signal categories (SLA breach + extreme metrics + cascade)
    if impact_result is not None and impact_result.n_categories_hit >= 3:
        composite = max(composite, _THRESHOLD_CRITICAL)
    
    # URGENCY OVERRIDE: High urgency should elevate severity regardless of impact
    # Urgency >= 0.85 alone can reach critical if sentiment is negative
    if urgency_score >= 0.85 and sentiment_label == "negative":
        composite = max(composite, _THRESHOLD_CRITICAL)
    
    # Urgency >= 0.75 alone can reach warning
    if urgency_score >= 0.75:
        composite = max(composite, _THRESHOLD_WARNING)

    # ── Map to severity ──
    severity, risk_level = _score_to_severity(composite)
    action_required = severity != "info"
    recommended_action = _build_action(severity, domain, impact_result)

    return SeverityResult(
        risk_level=risk_level,
        severity=severity,
        action_required=action_required,
        recommended_action=recommended_action,
    )


def _sentiment_to_weight(label: str) -> float:
    """Convert sentiment label to a [0, 1] weight for the composite."""
    if label == "negative":
        return 0.8
    if label == "neutral":
        return 0.3
    return 0.0  # positive


def _score_to_severity(composite: float) -> tuple:
    """Map composite score to (severity, risk_level)."""
    if composite >= _THRESHOLD_CRITICAL:
        return "critical", "HIGH"
    if composite >= _THRESHOLD_WARNING:
        return "warning", "MEDIUM"
    if composite >= 0.15:
        return "info", "LOW"
    return "info", "NONE"


def _build_action(
    severity: str,
    domain: str,
    impact: Optional[ImpactSignalResult],
) -> str:
    """Build human-readable recommended action."""
    domain_label = domain if domain != "general" else "document"

    if severity == "critical":
        base = (
            f"Critical issues detected in {domain_label}. "
            "Immediate review and action required."
        )
        if impact is not None and impact.summary:
            base += f" {impact.summary}"
        return base

    if severity == "warning":
        base = (
            f"Concerns identified in {domain_label}. "
            "Schedule review and monitor closely."
        )
        if impact is not None and impact.summary:
            base += f" {impact.summary}"
        return base

    return "No immediate action required. Continue standard monitoring."
