"""
MemoryEvent domain entity following DDD principles.

This is a value object representing operational memory events in the domain layer.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Mapping


@dataclass(frozen=True)
class MemoryEvent:
    """
    Domain entity for operational memory event (DDD value object).
    
    This represents the domain concept of an operational memory event,
    separate from infrastructure concerns.
    """
    
    series_id: int
    series_type: str
    timestamp: float
    event_type: str
    semantic_text: str
    regime: str
    anomaly_score: float
    dynamic_features: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def __init__(
        self,
        series_id: Optional[int] = None,
        series_type: Optional[str] = None,
        timestamp: float = 0.0,
        event_type: str = "observation",
        semantic_text: str = "",
        regime: str = "STABLE",
        anomaly_score: float = 0.0,
        dynamic_features: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        *,
        sensor_id: Optional[int] = None,
        sensor_type: Optional[str] = None,
    ) -> None:
        effective_series_id = series_id if series_id is not None else (sensor_id or 0)
        effective_series_type = series_type if series_type is not None else (sensor_type or "sensor")
        object.__setattr__(self, "series_id", effective_series_id)
        object.__setattr__(self, "series_type", effective_series_type)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "semantic_text", semantic_text)
        object.__setattr__(self, "regime", regime)
        object.__setattr__(self, "anomaly_score", anomaly_score)
        object.__setattr__(self, "dynamic_features", dynamic_features or {})
        object.__setattr__(self, "metadata", metadata or {})

    @property
    def sensor_id(self) -> int:
        return self.series_id

    @property
    def sensor_type(self) -> str:
        return self.series_type
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "series_id": self.series_id,
            "series_type": self.series_type,
            "sensor_id": self.series_id,
            "sensor_type": self.series_type,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "semantic_text": self.semantic_text,
            "regime": self.regime,
            "anomaly_score": self.anomaly_score,
            "dynamic_features": self.dynamic_features,
            "metadata": self.metadata,
        }