"""Sensor profile repository protocol."""

from __future__ import annotations

from typing import List, Optional, Protocol

from iot_machine_learning.domain.entities.sensor_profile import SensorProfile


class SensorProfileRepository(Protocol):
    """Protocol for loading sensor profiles."""

    def get_by_series_id(self, series_id: str) -> Optional[SensorProfile]: ...
    def get_all(self) -> List[SensorProfile]: ...
