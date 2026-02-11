"""Evaluador de umbrales de predicción — lógica de dominio pura.

Extraído de prediction_service.py._eval_pred_threshold_and_create_event().
Contiene SOLO reglas de decisión, sin I/O ni SQL.

Responsabilidad única: dado un valor predicho y un umbral,
determinar si hay violación y qué severidad tiene.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class ThresholdDefinition:
    """Definición de un umbral de alerta.

    Attributes:
        threshold_id: ID del umbral en BD.
        name: Nombre legible del umbral.
        condition_type: Tipo de condición.
        value_min: Valor mínimo (puede ser None).
        value_max: Valor máximo (puede ser None).
        severity: Severidad del umbral.
    """

    threshold_id: int
    name: str
    condition_type: str
    value_min: Optional[float]
    value_max: Optional[float]
    severity: str


@dataclass(frozen=True)
class ThresholdViolation:
    """Resultado de evaluación de umbral.

    Attributes:
        threshold_id: ID del umbral violado.
        threshold_name: Nombre del umbral.
        event_type: Tipo de evento a generar.
        title: Título del evento.
        message: Mensaje descriptivo.
        payload: Dict serializable con detalles.
    """

    threshold_id: int
    threshold_name: str
    event_type: str
    title: str
    message: str
    payload: dict


def is_threshold_violated(
    predicted_value: float,
    threshold: ThresholdDefinition,
) -> bool:
    """Evalúa si un valor predicho viola un umbral.

    Reglas de negocio puras — sin I/O.

    Args:
        predicted_value: Valor predicho.
        threshold: Definición del umbral.

    Returns:
        ``True`` si el valor viola el umbral.
    """
    cond = threshold.condition_type
    vmin = threshold.value_min
    vmax = threshold.value_max

    if cond == "greater_than" and vmin is not None:
        return predicted_value > vmin
    if cond == "less_than" and vmin is not None:
        return predicted_value < vmin
    if cond == "out_of_range" and vmin is not None and vmax is not None:
        return predicted_value < vmin or predicted_value > vmax
    if cond == "equal_to" and vmin is not None:
        return predicted_value == vmin

    return False


def is_within_warning_range(
    value: float,
    warning_min: Optional[float],
    warning_max: Optional[float],
) -> bool:
    """Verifica si un valor está dentro del rango WARNING del usuario.

    Regla de dominio: si está dentro del rango, ML NO debe alertar.

    Args:
        value: Valor a verificar.
        warning_min: Límite inferior (None = sin límite).
        warning_max: Límite superior (None = sin límite).

    Returns:
        ``True`` si el valor está dentro del rango.
    """
    if warning_min is None and warning_max is None:
        return False

    if warning_min is not None and value < warning_min:
        return False
    if warning_max is not None and value > warning_max:
        return False

    return True


def build_violation(
    predicted_value: float,
    threshold: ThresholdDefinition,
) -> ThresholdViolation:
    """Construye un ThresholdViolation a partir de un umbral violado.

    Args:
        predicted_value: Valor predicho que viola el umbral.
        threshold: Definición del umbral violado.

    Returns:
        ``ThresholdViolation`` con todos los datos para persistir el evento.
    """
    sev = str(threshold.severity)
    if sev == "critical":
        event_type = "critical"
    elif sev == "warning":
        event_type = "warning"
    else:
        event_type = "notice"

    title = f"Predicción viola umbral: {threshold.name}"
    message = (
        f"predicted_value={predicted_value} "
        f"threshold_id={threshold.threshold_id}"
    )

    payload = {
        "threshold_id": threshold.threshold_id,
        "condition_type": threshold.condition_type,
        "threshold_value_min": threshold.value_min,
        "threshold_value_max": threshold.value_max,
        "predicted_value": predicted_value,
    }

    return ThresholdViolation(
        threshold_id=threshold.threshold_id,
        threshold_name=threshold.name,
        event_type=event_type,
        title=title,
        message=message,
        payload=payload,
    )
