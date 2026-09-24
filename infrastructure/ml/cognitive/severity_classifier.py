"""Severity classifier for sensor predictions in ML Core.

Combines statistical anomalies with physical risk rules.
Strictly decoupled from databases, SQL, and external protocols.
Delegates threshold lookups to an injected ThresholdProvider.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Protocol, Tuple, runtime_checkable

from iot_machine_learning.domain.services.severity_rules import (
    SeverityResult,
    classify_severity,
    compute_risk_level,
    compute_severity,
    is_out_of_range,
)
from iot_machine_learning.domain.entities.iot.sensor_ranges import get_default_range

logger = logging.getLogger(__name__)

__all__ = ["SeverityClassifier", "SeverityResult", "ThresholdProvider"]


@runtime_checkable
class ThresholdProvider(Protocol):
    """Protocol for external sources providing sensor alert thresholds."""

    def get_user_defined_range(
        self,
        conn: Any,
        sensor_id: int,
    ) -> Optional[Tuple[float, float]]:
        """Retrieves user-defined threshold range (min, max)."""
        ...

    def is_value_within_user_thresholds(
        self,
        conn: Any,
        sensor_id: int,
        value: float,
    ) -> bool:
        """Checks if value is within user-defined warning range."""
        ...


class SeverityClassifier:
    """Classifies prediction severity combining statistical and physical risk.

    Pure mathematical and cognitive evaluation. Accepts an optional ThresholdProvider
    for retrieving external user threshold configurations.
    """

    def __init__(
        self,
        threshold_provider: Optional[ThresholdProvider] = None,
    ) -> None:
        self._threshold_provider = threshold_provider

    def get_user_defined_range(
        self,
        conn: Any,
        sensor_id: int,
    ) -> Optional[Tuple[float, float]]:
        """Retrieves user-defined range via injected provider if available."""
        if self._threshold_provider is not None:
            return self._threshold_provider.get_user_defined_range(conn, sensor_id)
        return None

    def is_value_within_user_thresholds(
        self,
        conn: Any,
        sensor_id: int,
        value: float,
    ) -> bool:
        """Checks threshold validity via injected provider if available."""
        if self._threshold_provider is not None:
            return self._threshold_provider.is_value_within_user_thresholds(
                conn, sensor_id, value
            )
        return True

    def classify(
        self,
        *,
        sensor_type: str,
        location: str,
        predicted_value: float,
        trend: str,
        anomaly: bool,
        anomaly_score: float,
        confidence: float,
        horizon_minutes: int,
        user_defined_range: Optional[Tuple[float, float]] = None,
    ) -> SeverityResult:
        """Evaluates prediction severity against domain rules."""
        return classify_severity(
            sensor_type=sensor_type,
            location=location,
            predicted_value=predicted_value,
            anomaly=anomaly,
            user_defined_range=user_defined_range,
        )
