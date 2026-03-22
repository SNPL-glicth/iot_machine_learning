"""Explanation building using domain Explanation + ExplanationRenderer.

Constructs ``Explanation`` value objects and uses ``ExplanationRenderer``
to produce human-readable output.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy imports — explainability layer may not be importable
_explainability_available = True
try:
    from iot_machine_learning.domain.entities.explainability.explanation import Explanation, Outcome
    from iot_machine_learning.domain.entities.explainability.signal_snapshot import SignalSnapshot
    from iot_machine_learning.application.explainability.explanation_renderer import ExplanationRenderer

    _renderer = ExplanationRenderer()
except Exception as exc:
    _explainability_available = False
    logger.warning("[CONCLUSION_BUILDER] Explainability layer not available: %s", exc)


def build_text_explanation(
    *,
    document_id: str,
    sentiment_label: str,
    sentiment_score: float,
    urgency_score: float,
    urgency_severity: str,
    urgency_hits: List[Dict[str, Any]],
    readability_avg_sentence_len: float,
    readability_vocabulary_richness: float,
    structural_regime: str = "unknown",
    structural_trend: float = 0.0,
    structural_noise: float = 0.0,
    confidence: float = 0.75,
) -> Optional[Dict[str, Any]]:
    """Build an Explanation value object from text analysis results.

    Returns the explanation dict or None if explainability is unavailable.
    """
    if not _explainability_available:
        return None

    try:
        signal = SignalSnapshot(
            n_points=0,
            mean=urgency_score,
            std=0.0,
            noise_ratio=structural_noise,
            slope=structural_trend,
            regime=structural_regime,
            extra={
                "sentiment": sentiment_label,
                "sentiment_score": sentiment_score,
                "urgency_score": urgency_score,
                "urgency_severity": urgency_severity,
                "vocabulary_richness": readability_vocabulary_richness,
                "avg_sentence_length": readability_avg_sentence_len,
            },
        )

        outcome = Outcome(
            kind="text_analysis",
            confidence=confidence,
            trend=structural_regime,
            is_anomaly=urgency_severity == "critical",
            anomaly_score=urgency_score if urgency_severity == "critical" else None,
            extra={
                "sentiment": sentiment_label,
                "urgency_severity": urgency_severity,
            },
        )

        explanation = Explanation(
            series_id=document_id,
            signal=signal,
            outcome=outcome,
        )

        return explanation.to_dict()
    except Exception as exc:
        logger.warning("[CONCLUSION_BUILDER] Failed to build explanation: %s", exc)
        return None
