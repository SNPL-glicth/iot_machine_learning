"""Sensor profile entity with equipment-aware calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from iot_machine_learning.domain.value_objects.equipment_class import EquipmentClass


@dataclass(frozen=True)
class SensorProfile:
    """Equipment-aware sensor calibration profile."""

    series_id: str
    equipment_class: EquipmentClass
    operational_range: Tuple[float, float]
    setpoint_tolerance: float
    noise_floor: float
    maintenance_history_score: float
    hampel_k: float = 3.0
    hampel_window: int = 10
    typical_cycle_duration_seconds: Optional[int] = None
    thermal_inertia_class: Optional[Literal["fast", "medium", "slow"]] = None

    def is_value_in_range(self, value: float) -> bool:
        """Check if value is within operational range."""
        return self.operational_range[0] <= value <= self.operational_range[1]

    def relative_deviation(self, current_std: float) -> float:
        """Compute relative deviation from noise floor."""
        return (current_std - self.noise_floor) / max(self.setpoint_tolerance, 1e-9)
