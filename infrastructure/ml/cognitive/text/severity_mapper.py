"""Wrapper de adaptación entre UniversalAnalysisEngine y ThresholdPolicy.

Traduce los parámetros que engine.py extrae de las percepciones
(text_urgency, text_sentiment) al protocolo de classify_text().

Decisiones de diseño (no reconstrucción del original eliminado):
- sentiment_label → sentiment_weight via _SENTIMENT_WEIGHT_MAP
- impact_score = 0.0 porque engine.py nunca provee impacto
- urgency_override = has_critical_keywords (1:1)
"""

from __future__ import annotations

from typing import Any

from iot_machine_learning.domain.entities.severity import SeverityResult
from iot_machine_learning.domain.policies.threshold_policy import ThresholdPolicy

_SENTIMENT_WEIGHT_MAP: dict[str, float] = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral": 0.0,
}
_DEFAULT_IMPACT_SCORE: float = 0.0


def classify_text_severity(
    urgency_score: float,
    urgency_severity: str,
    sentiment_label: str,
    has_critical_keywords: bool,
    domain: str,
    full_text: str = "",
    impact_result: Any = None,
) -> SeverityResult:
    """Adaptación: parámetros engine.py → ThresholdPolicy.classify_text().

    Args:
        urgency_score: Score de urgencia [0, 1].
        urgency_severity: "info" | "warning" | "critical".
        sentiment_label: "positive" | "negative" | "neutral".
        has_critical_keywords: Flag de keywords críticas.
        domain: Dominio del texto.
        full_text: Texto completo (no usado por engine.py, provisto por compatibilidad).
        impact_result: Resultado de análisis de impacto (no usado, siempre None).

    Returns:
        SeverityResult con risk_level, severity, action_required, recommended_action.
    """
    sentiment_weight = _SENTIMENT_WEIGHT_MAP.get(sentiment_label, 0.0)
    urgency_override = has_critical_keywords

    policy_result = ThresholdPolicy.default().classify_text(
        urgency_score=urgency_score,
        sentiment_weight=sentiment_weight,
        impact_score=_DEFAULT_IMPACT_SCORE,
        domain=domain,
        n_categories_hit=0,
        urgency_override=urgency_override,
    )

    return SeverityResult(
        risk_level=policy_result.risk_level,
        severity=policy_result.severity_label,
        action_required=policy_result.action_required,
        recommended_action=policy_result.recommended_action,
    )
