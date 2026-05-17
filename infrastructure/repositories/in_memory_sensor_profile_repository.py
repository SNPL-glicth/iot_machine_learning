"""In-memory sensor profile repository for testing."""

from __future__ import annotations

from typing import Dict, List, Optional

from iot_machine_learning.domain.entities.sensor_profile import SensorProfile
from iot_machine_learning.domain.value_objects.equipment_class import EquipmentClass


def _default(series_id: str) -> SensorProfile:
    return SensorProfile(
        series_id=series_id, equipment_class=EquipmentClass.GENERIC,
        operational_range=(0.0, 100.0), setpoint_tolerance=5.0,
        noise_floor=1.0, maintenance_history_score=0.5,
    )


class InMemorySensorProfileRepository:
    """Test double for SensorProfileRepository."""

    def __init__(self, profiles: Dict[str, SensorProfile]) -> None:
        self._profiles = dict(profiles)

    def get_by_series_id(self, series_id: str) -> Optional[SensorProfile]:
        return self._profiles.get(series_id, _default(series_id))

    def get_all(self) -> List[SensorProfile]:
        return list(self._profiles.values())
