"""Legacy severity computation functions (DEPRECATED).

These functions are preserved for backward compatibility.
New code should use ``ThresholdPolicy`` directly.
"""

from __future__ import annotations

import warnings
from typing import Optional

from ...entities.severity import SeverityResult
from ...entities.threshold import Threshold
from .severity_helpers import (
    build_recommended_action,
    compute_risk_level_from_threshold,
)


def compute_risk_level(
    value: float,
    threshold: Optional[Threshold],
) -> str:
    """Legacy alias for compute_risk_level_from_threshold."""
    return compute_risk_level_from_threshold(value, threshold)


def is_out_of_range(
    value: float,
    threshold: Optional[Threshold],
) -> bool:
    """Check if value is out of physical range."""
    if threshold is None:
        return False
    severity = threshold.severity_for(value)
    return severity == "critical"


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
    from ...policies.threshold_policy import ThresholdPolicy

    result = ThresholdPolicy.default().classify_with_context(
        score=0.5,
        is_anomaly=is_anomaly,
        risk_level=risk_level,
        out_of_physical_range=out_of_physical_range,
    )
    return result.severity_label


def classify_severity(
    sensor_type: str,
    value: float,
    anomaly: bool,
    category: Optional[str] = None,
    threshold: Optional[Threshold] = None,
    label: str = "",
) -> SeverityResult:
    """Legacy sensor-type-aware severity classification (DEPRECATED)."""
    warnings.warn(
        "classify_severity(sensor_type) is deprecated. "
        "Use classify_severity_agnostic() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if threshold is not None:
        risk_level = compute_risk_level(value, threshold)
    else:
        risk_level = "normal"

    severity = compute_severity(
        is_anomaly=anomaly,
        risk_level=risk_level,
        out_of_physical_range=False,
    )

    action_required = severity in {"critical", "warning"}
    return SeverityResult(
        risk_level=risk_level,
        severity=severity,
        action_required=action_required,
        recommended_action=build_recommended_action(severity),
    )
