"""TextSeverityMapper — 3-axis severity classification.

Combines three independent axes to compute real-world severity.

Produces the same ``SeverityResult`` dataclass used by the IoT pipeline
(``domain.services.severity_rules``) but without requiring the
IoT-specific ``Threshold`` entity.

No imports from ml_service — only domain layer + sibling modules.
Single entry point: ``classify_text_severity()``.

All thresholds and weights now live in ``ThresholdPolicy`` (single
source of truth).  This module translates text-specific signals into
policy inputs.
"""

from __future__ import annotations

from typing import Optional

from iot_machine_learning.domain.policies.threshold_policy import ThresholdPolicy
from iot_machine_learning.domain.services.severity_rules import SeverityResult

from .impact_detector import ImpactSignalResult, detect_impact_signals


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

    Delegates to ``ThresholdPolicy.classify_text()`` for unified
    severity logic (single source of truth).

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

    # ── Unified classification via ThresholdPolicy ──
    policy = ThresholdPolicy.default()
    result = policy.classify_text(
        urgency_score=urgency_score,
        sentiment_weight=sentiment_weight,
        impact_score=impact_score,
        domain=domain,
        n_categories_hit=impact_result.n_categories_hit if impact_result else 0,
        urgency_override=(
            urgency_score >= 0.85 and sentiment_label == "negative"
        ) or urgency_score >= 0.75,
    )

    # Append impact summary to action when available
    recommended_action = result.recommended_action
    if impact_result is not None and impact_result.summary:
        if result.severity_label in {"critical", "warning"}:
            recommended_action += f" {impact_result.summary}"

    return SeverityResult(
        risk_level=result.risk_level,
        severity=result.severity_label,
        action_required=result.action_required,
        recommended_action=recommended_action,
    )


def _sentiment_to_weight(label: str) -> float:
    """Convert sentiment label to a [0, 1] weight for the composite."""
    if label == "negative":
        return 0.8
    if label == "neutral":
        return 0.3
    return 0.0  # positive
