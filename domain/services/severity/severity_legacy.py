"""Legacy severity computation functions (DEPRECATED).

These functions are preserved for backward compatibility.
New code should use ``ThresholdPolicy`` directly.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple, Union

from ...entities.severity import SeverityResult
from ...entities.threshold import Threshold
from ...entities.sensor_ranges import get_default_range
from .severity_helpers import compute_risk_level_from_threshold


def compute_risk_level(
    sensor_type_or_value: Union[str, float],
    value_or_threshold: Any = None,
    user_defined_range: Optional[Tuple[float, float]] = None,
) -> str:
    """Compute risk level from sensor_type + value, or value + Threshold."""
    # Case 1: Legacy IoT sensor-type call: compute_risk_level(sensor_type: str, value: float, user_range=...)
    if isinstance(sensor_type_or_value, str):
        sensor_type = sensor_type_or_value
        value = float(value_or_threshold) if value_or_threshold is not None else 0.0
        range_tuple = user_defined_range or get_default_range(sensor_type)
        if range_tuple is None:
            return "NONE"
        min_val, max_val = range_tuple
        if min_val <= value <= max_val:
            return "LOW"
        margin = 0.1 * (max_val - min_val)
        if (min_val - margin) <= value <= (max_val + margin):
            return "MEDIUM"
        return "HIGH"

    # Case 2: Agnostic threshold call: compute_risk_level(value: float, threshold: Optional[Threshold])
    value = float(sensor_type_or_value)
    threshold = value_or_threshold
    return compute_risk_level_from_threshold(value, threshold)


def is_out_of_range(
    value: float,
    range_or_threshold: Optional[Union[Tuple[float, float], Threshold]],
) -> bool:
    """Check if value is out of physical range (Tuple or Threshold)."""
    if range_or_threshold is None:
        return False
    if isinstance(range_or_threshold, tuple):
        min_val, max_val = range_or_threshold
        return value < min_val or value > max_val
    return range_or_threshold.severity_for(value) == "critical"


def compute_severity(
    *,
    is_anomaly: bool,
    risk_level: str,
    out_of_physical_range: bool,
) -> str:
    """DEPRECATED: Compute severity label from flags."""
    warnings.warn(
        "compute_severity() is deprecated. Use ThresholdPolicy.classify_with_context() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if out_of_physical_range:
        return "critical"
    rl = risk_level.upper()
    if is_anomaly and rl == "HIGH":
        return "critical"
    if is_anomaly or rl == "HIGH":
        return "warning"
    return "info"


def build_recommended_action(
    severity: str,
    risk_level: str = "LOW",
    location: str = "",
) -> str:
    """Build recommended action narrative based on severity, risk and location."""
    loc_str = f" en {location}" if location else ""
    sev = severity.lower()
    rl = risk_level.upper()

    if sev == "critical":
        return f"Alerta crítica{loc_str}: valor fuera de límites operativos. Requiere atención inmediata."
    if sev == "warning" and rl == "HIGH":
        return f"Riesgo elevado{loc_str}: valor cercano a límites operativos. Se recomienda inspección preventiva."
    if sev == "warning":
        return f"Comportamiento inusual{loc_str}: anomalía detectada. Monitorear de cerca."
    if sev == "info" and rl == "MEDIUM":
        return f"Advertencia menor{loc_str}: valor dentro de límites operativos pero con leve desviación."
    return "No se requiere acción. Comportamiento dentro de lo esperado."


def classify_severity(
    sensor_type: str = "",
    value: Optional[float] = None,
    anomaly: bool = False,
    category: Optional[str] = None,
    threshold: Optional[Threshold] = None,
    label: str = "",
    location: str = "",
    predicted_value: Optional[float] = None,
    user_defined_range: Optional[Tuple[float, float]] = None,
    **kwargs: Any,
) -> SeverityResult:
    """Legacy sensor-type-aware severity classification (DEPRECATED)."""
    warnings.warn(
        "classify_severity(sensor_type) is deprecated. "
        "Use classify_severity_agnostic() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    val = predicted_value if predicted_value is not None else (value if value is not None else 0.0)

    # If threshold is provided and no sensor_type/user_range, use agnostic calculation
    if threshold is not None and not user_defined_range:
        risk_level = compute_risk_level_from_threshold(val, threshold)
        out_of_range = threshold.severity_for(val) == "critical"
    else:
        risk_level = compute_risk_level(sensor_type, val, user_defined_range=user_defined_range)
        effective_range = user_defined_range or get_default_range(sensor_type)
        out_of_range = is_out_of_range(val, effective_range)

    severity = compute_severity(
        is_anomaly=anomaly,
        risk_level=risk_level,
        out_of_physical_range=out_of_range,
    )

    action_required = severity in {"critical", "warning"}
    return SeverityResult(
        risk_level=risk_level,
        severity=severity,
        action_required=action_required,
        recommended_action=build_recommended_action(
            severity=severity,
            risk_level=risk_level,
            location=location,
        ),
    )
