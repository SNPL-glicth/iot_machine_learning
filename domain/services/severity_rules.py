"""Reglas de severidad — lógica de dominio pura.

Extraído de severity_classifier.py.
Responsabilidad ÚNICA: dado un valor predicho, tipo de sensor y estado de anomalía,
determinar nivel de riesgo, severidad y acción recomendada.

No contiene I/O (no lee BD, no persiste).
No depende de sqlalchemy ni de ningún framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from ..entities.sensor_ranges import get_default_range


@dataclass(frozen=True)
class SeverityResult:
    """Resultado de clasificación de severidad.

    Value object puro — sin lógica de negocio.
    """

    risk_level: str
    severity: str
    action_required: bool
    recommended_action: str


def compute_risk_level(
    sensor_type: str,
    predicted_value: float,
) -> str:
    """Clasifica nivel de riesgo físico basado en rangos del sensor.

    Regla de dominio pura — sin I/O.

    Args:
        sensor_type: Tipo de sensor (e.g. "temperature").
        predicted_value: Valor predicho.

    Returns:
        ``'LOW'`` | ``'MEDIUM'`` | ``'HIGH'`` | ``'NONE'``
    """
    rng = get_default_range(sensor_type)
    if rng is None:
        return "NONE"

    min_ok, max_ok = rng
    if min_ok <= predicted_value <= max_ok:
        return "LOW"

    margin = 0.1 * (max_ok - min_ok)
    if predicted_value < min_ok - margin or predicted_value > max_ok + margin:
        return "HIGH"

    return "MEDIUM"


def compute_severity(
    *,
    is_anomaly: bool,
    risk_level: str,
    out_of_physical_range: bool,
) -> str:
    """Combina anomalía + riesgo en severidad única.

    Regla de dominio pura — sin I/O.

    Prioridad:
    1. Fuera de rango físico → critical
    2. Anomalía + riesgo HIGH → critical
    3. Anomalía o riesgo HIGH → warning
    4. Resto → info

    Args:
        is_anomaly: Si se detectó anomalía estadística.
        risk_level: Nivel de riesgo físico.
        out_of_physical_range: Si el valor está fuera del rango físico.

    Returns:
        ``'critical'`` | ``'warning'`` | ``'info'``
    """
    rl = (risk_level or "").upper()

    if out_of_physical_range:
        return "critical"
    if is_anomaly and rl == "HIGH":
        return "critical"
    if is_anomaly or rl == "HIGH":
        return "warning"
    return "info"


def is_out_of_range(
    value: float,
    range_bounds: Optional[Tuple[float, float]],
) -> bool:
    """Verifica si un valor está fuera de un rango dado.

    Args:
        value: Valor a verificar.
        range_bounds: Tupla (min, max) o None.

    Returns:
        ``True`` si fuera de rango, ``False`` si dentro o sin rango.
    """
    if range_bounds is None:
        return False
    min_ok, max_ok = range_bounds
    return value < min_ok or value > max_ok


def build_recommended_action(
    *,
    severity: str,
    risk_level: str,
    location: str,
) -> str:
    """Genera acción recomendada según severidad y riesgo.

    Regla de narrativa de dominio — sin I/O.

    Args:
        severity: Severidad calculada.
        risk_level: Nivel de riesgo físico.
        location: Ubicación del sensor.

    Returns:
        Texto de acción recomendada.
    """
    rl = risk_level.upper()

    if severity == "info":
        if rl in {"MEDIUM", "HIGH"}:
            return (
                f"La proyección se acerca a los límites operativos en {location}. "
                "Supervisar en las próximas horas."
            )
        return "Valores dentro del rango esperado. No se requiere acción."

    if severity == "critical":
        return (
            f"Condición crítica en {location}. "
            "Revisar inmediatamente el equipo y condiciones ambientales."
        )

    # warning
    if rl == "HIGH":
        return (
            f"Riesgo elevado en {location}. "
            "Programar revisión prioritaria."
        )
    return (
        f"Comportamiento inusual en {location}. "
        "Supervisar de cerca."
    )


def classify_severity(
    *,
    sensor_type: str,
    location: str,
    predicted_value: float,
    anomaly: bool,
    user_defined_range: Optional[Tuple[float, float]] = None,
) -> SeverityResult:
    """Clasificación completa de severidad — función de dominio pura.

    Combina riesgo físico, anomalía estadística y rangos del usuario.

    Args:
        sensor_type: Tipo de sensor.
        location: Ubicación del sensor.
        predicted_value: Valor predicho.
        anomaly: Si se detectó anomalía.
        user_defined_range: Rango definido por el usuario (prioridad sobre default).

    Returns:
        ``SeverityResult`` con risk_level, severity, action_required, recommended_action.
    """
    risk_level = compute_risk_level(sensor_type, predicted_value)

    # Rango efectivo: usuario > default
    effective_range = user_defined_range or get_default_range(sensor_type)
    out_of_range = is_out_of_range(predicted_value, effective_range)

    severity = compute_severity(
        is_anomaly=anomaly,
        risk_level=risk_level,
        out_of_physical_range=out_of_range,
    )

    action_required = severity != "info"

    recommended_action = build_recommended_action(
        severity=severity,
        risk_level=risk_level,
        location=location,
    )

    return SeverityResult(
        risk_level=risk_level,
        severity=severity,
        action_required=action_required,
        recommended_action=recommended_action,
    )
