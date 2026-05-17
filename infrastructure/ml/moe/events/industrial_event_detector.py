"""Industrial event detector — pure function, no state, no I/O."""

from __future__ import annotations

import logging
from typing import List, Optional

from iot_machine_learning.domain.value_objects.equipment_class import EquipmentClass
from iot_machine_learning.domain.value_objects.industrial_event import (
    EventContext,
    IndustrialEvent,
)

logger = logging.getLogger(__name__)


def detect_industrial_event(
    values: List[float],
    sanitization_flags: List[str],
    sensor_profile: Optional[object] = None,
) -> EventContext:
    """Detecta evento industrial basado en señal y flags del pipeline."""
    try:
        noise_floor = getattr(sensor_profile, "noise_floor", 1.0)
        eq = getattr(sensor_profile, "equipment_class", None)

        if "cusum_ramp_detected" in sanitization_flags and values:
            mag = abs(values[-1] - values[0])
            if eq in (EquipmentClass.PASTEURIZER, EquipmentClass.CIP):
                if mag > 3 * noise_floor:
                    return EventContext(IndustrialEvent.STARTUP, 0, 0.8)
                if mag > noise_floor:
                    return EventContext(IndustrialEvent.CIP_CYCLE, 0, 0.7)
            if eq == EquipmentClass.SILO:
                return EventContext(IndustrialEvent.PRODUCT_CHANGEOVER, 0, 0.6)
            return EventContext(IndustrialEvent.UNKNOWN, 0, 0.5)

        clamped = sum(
            int(f.split(":")[1])
            for f in sanitization_flags
            if f.startswith("value_clamped:")
        )
        if clamped > 0 and values and clamped / len(values) > 0.30:
            return EventContext(IndustrialEvent.FAULT_TRANSIENT, 0, 0.6)

        if len(values) >= 3:
            delta = abs(values[-1] - values[-2])
            if sensor_profile is not None and delta > 5 * noise_floor:
                return EventContext(IndustrialEvent.STARTUP, 0, 0.5)
            if sensor_profile is None and delta > 10.0:
                return EventContext(IndustrialEvent.UNKNOWN, 0, 0.4)

        return EventContext.none()
    except Exception:
        return EventContext.none()
