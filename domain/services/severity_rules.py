"""Reglas de severidad — lógica de dominio pura.

Responsabilidad ÚNICA: dado un valor predicho y estado de anomalía,
determinar nivel de riesgo, severidad y acción recomendada.
"""

from __future__ import annotations

from typing import Optional

from ..entities.severity import SeverityResult
from ..entities.threshold import Threshold

# Re-export helpers so existing imports keep working.
from .severity_helpers import (
    action_for_severity,
    build_recommended_action,
    compute_risk_level_from_threshold,
    severity_from_risk,
)
from .severity_legacy import (
    compute_risk_level,
    compute_severity,
    is_out_of_range,
    classify_severity,
)

__all__ = [
    "action_for_severity",
    "build_recommended_action",
    "classify_severity",
    "classify_severity_agnostic",
    "compute_risk_level",
    "compute_risk_level_from_threshold",
    "compute_severity",
    "is_out_of_range",
    "severity_from_risk",
]

# ---------------------------------------------------------------------------
# Funciones agnósticas (Nivel 2 UTSAE) — usan Threshold en vez de sensor_type
# ---------------------------------------------------------------------------


def classify_severity_agnostic(
    *,
    value: float,
    anomaly: bool,
    threshold: Optional[Threshold] = None,
    label: str = "",
) -> SeverityResult:
    """Clasificación de severidad agnóstica al dominio.

    Delegates to ``ThresholdPolicy`` for unified severity classification.

    Args:
        value: Valor predicho o actual.
        anomaly: Si se detectó anomalía estadística.
        threshold: Umbral configurado (de ``SeriesContext``).
        label: Etiqueta descriptiva para la narrativa (e.g. "serie X").

    Returns:
        ``SeverityResult`` con risk_level, severity, action y recommended_action.
    """
    from ..policies.threshold_policy import ThresholdPolicy

    risk_level = compute_risk_level_from_threshold(value, threshold)

    out_of_range = False
    if threshold is not None:
        sev = threshold.severity_for(value)
        out_of_range = sev == "critical"

    result = ThresholdPolicy.default().classify_with_context(
        score=0.5,
        is_anomaly=anomaly,
        risk_level=risk_level,
        out_of_physical_range=out_of_range,
        label=label,
    )

    return SeverityResult(
        risk_level=result.risk_level,
        severity=result.severity_label,
        action_required=result.action_required,
        recommended_action=result.recommended_action,
    )
