"""IndustrialEvent value objects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class IndustrialEvent(str, Enum):
    """Industrial events detected in sensor signals."""

    STARTUP = "STARTUP"
    SHUTDOWN = "SHUTDOWN"
    CIP_CYCLE = "CIP_CYCLE"
    PRODUCT_CHANGEOVER = "PRODUCT_CHANGEOVER"
    FAULT_TRANSIENT = "FAULT_TRANSIENT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class EventContext:
    """Evento industrial detectado en la señal actual."""

    detected_event: Optional[IndustrialEvent]
    time_since_event_seconds: Optional[int]
    event_confidence: float
    source: str = "detector"

    @property
    def is_active(self) -> bool:
        """True si hay un evento activo con confianza > 0.5."""
        return self.detected_event is not None and self.event_confidence > 0.5

    @classmethod
    def none(cls) -> "EventContext":
        """Factory para cuando no hay evento detectado."""
        return cls(
            detected_event=None,
            time_since_event_seconds=None,
            event_confidence=0.0,
        )
