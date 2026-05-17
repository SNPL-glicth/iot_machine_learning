"""Tests for sensor profile repository implementations."""

from __future__ import annotations

from iot_machine_learning.domain.entities.sensor_profile import SensorProfile
from iot_machine_learning.domain.value_objects.equipment_class import EquipmentClass
from iot_machine_learning.infrastructure.repositories.in_memory_sensor_profile_repository import (
    InMemorySensorProfileRepository,
)


def test_from_device_type_pasteurizer() -> None:
    assert EquipmentClass.from_device_type("PASTEUR") == EquipmentClass.PASTEURIZER


def test_from_device_type_unknown_falls_back_to_generic() -> None:
    assert EquipmentClass.from_device_type("XYZ_DESCONOCIDO") == EquipmentClass.GENERIC


def test_get_by_series_id_returns_profile() -> None:
    p = SensorProfile(
        series_id="42", equipment_class=EquipmentClass.FILLER,
        operational_range=(0.0, 10.0), setpoint_tolerance=0.2,
        noise_floor=0.05, maintenance_history_score=0.5,
        hampel_k=4.0, hampel_window=20,
    )
    repo = InMemorySensorProfileRepository({"42": p})
    assert repo.get_by_series_id("42") == p


def test_get_by_series_id_missing_returns_generic() -> None:
    repo = InMemorySensorProfileRepository({})
    result = repo.get_by_series_id("999")
    assert result is not None
    assert result.equipment_class == EquipmentClass.GENERIC


def test_relative_deviation_calculation() -> None:
    p = SensorProfile(
        series_id="1", equipment_class=EquipmentClass.GENERIC,
        operational_range=(0.0, 100.0), setpoint_tolerance=2.0,
        noise_floor=0.5, maintenance_history_score=0.5,
    )
    assert p.relative_deviation(1.5) == 0.5


def test_is_value_in_range() -> None:
    p = SensorProfile(
        series_id="1", equipment_class=EquipmentClass.GENERIC,
        operational_range=(60.0, 80.0), setpoint_tolerance=1.0,
        noise_floor=0.5, maintenance_history_score=0.5,
    )
    assert p.is_value_in_range(72.0) is True
    assert p.is_value_in_range(85.0) is False
