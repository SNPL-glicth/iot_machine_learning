"""Tipos y entidades para correlación de sensores."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class CorrelationPattern(Enum):
    """Patrones de correlación conocidos entre sensores."""
    HVAC_FAILURE = "hvac_failure"           # temp↑ + humedad↓
    HVAC_OVERLOAD = "hvac_overload"         # temp↑ + power↑
    WATER_LEAK = "water_leak"               # humedad↑ + temp↓
    ELECTRICAL_ISSUE = "electrical_issue"   # voltage↓ + power fluctuation
    VENTILATION_BLOCKED = "ventilation_blocked"  # temp↑ + air_quality↓
    UNKNOWN = "unknown"


@dataclass
class SensorSnapshot:
    """Snapshot del estado actual de un sensor."""
    sensor_id: int
    sensor_type: str
    current_value: float
    predicted_value: float
    trend: str  # rising, falling, stable
    anomaly_score: float
    severity: str
    timestamp: datetime


@dataclass
class CorrelationResult:
    """Resultado de análisis de correlación entre sensores."""
    device_id: int
    device_name: str
    pattern_detected: Optional[CorrelationPattern]
    pattern_confidence: float
    correlated_sensors: list[SensorSnapshot]
    combined_severity: str
    description: str
    root_cause_hypothesis: Optional[str]
    is_significant: bool  # True si la correlación es significativa
    
    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "pattern_detected": self.pattern_detected.value if self.pattern_detected else None,
            "pattern_confidence": self.pattern_confidence,
            "correlated_sensors": [
                {
                    "sensor_id": s.sensor_id,
                    "sensor_type": s.sensor_type,
                    "current_value": s.current_value,
                    "predicted_value": s.predicted_value,
                    "trend": s.trend,
                    "anomaly_score": s.anomaly_score,
                    "severity": s.severity,
                }
                for s in self.correlated_sensors
            ],
            "combined_severity": self.combined_severity,
            "description": self.description,
            "root_cause_hypothesis": self.root_cause_hypothesis,
            "is_significant": self.is_significant,
        }


@dataclass
class DeviceSensorGroup:
    """Grupo de sensores de un dispositivo."""
    device_id: int
    device_name: str
    sensors: list[SensorSnapshot] = field(default_factory=list)
