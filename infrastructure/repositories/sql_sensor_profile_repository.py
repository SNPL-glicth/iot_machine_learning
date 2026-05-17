"""SQL-backed sensor profile repository with in-memory cache."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.engine import Connection

from iot_machine_learning.domain.entities.sensor_profile import SensorProfile
from iot_machine_learning.domain.value_objects.equipment_class import EquipmentClass

logger = logging.getLogger(__name__)

EQUIPMENT_DEFAULTS: Dict[EquipmentClass, Dict[str, Any]] = {
    EquipmentClass.PASTEURIZER: {"noise_floor": 0.3, "setpoint_tolerance": 0.5, "hampel_k": 2.5, "hampel_window": 10, "operational_range": (60.0, 85.0)},
    EquipmentClass.CIP: {"noise_floor": 1.5, "setpoint_tolerance": 5.0, "hampel_k": 5.0, "hampel_window": 5, "operational_range": (0.0, 100.0)},
    EquipmentClass.FILLER: {"noise_floor": 0.05, "setpoint_tolerance": 0.2, "hampel_k": 4.0, "hampel_window": 20, "operational_range": (0.0, 10.0)},
    EquipmentClass.PET_BLOWER: {"noise_floor": 0.5, "setpoint_tolerance": 2.0, "hampel_k": 4.5, "hampel_window": 8, "operational_range": (0.0, 50.0)},
    EquipmentClass.CONVEYOR: {"noise_floor": 0.2, "setpoint_tolerance": 1.0, "hampel_k": 4.5, "hampel_window": 15, "operational_range": (0.0, 100.0)},
    EquipmentClass.SILO: {"noise_floor": 0.5, "setpoint_tolerance": 3.0, "hampel_k": 3.0, "hampel_window": 30, "operational_range": (0.0, 100.0)},
    EquipmentClass.GENERIC: {"noise_floor": 1.0, "setpoint_tolerance": 5.0, "hampel_k": 3.0, "hampel_window": 10, "operational_range": (0.0, 100.0)},
}


class SqlSensorProfileRepository:
    """Loads SensorProfile from dbo.sensors + dbo.devices with EquipmentClass defaults."""

    def __init__(self, conn: Connection) -> None:
        self._conn = conn
        self._cache: Dict[str, SensorProfile] = {}

    def get_by_series_id(self, series_id: str) -> Optional[SensorProfile]:
        if series_id in self._cache:
            return self._cache[series_id]
        profile = self._load(series_id)
        if profile:
            self._cache[series_id] = profile
        return profile

    def get_all(self) -> List[SensorProfile]:
        return list(self._cache.values())

    def _load(self, series_id: str) -> Optional[SensorProfile]:
        try:
            sensor_id = int(series_id)
        except ValueError:
            return self._default(series_id)
        try:
            row = self._conn.execute(text("""
                SELECT TOP 1 COALESCE(d.device_type, '') AS dt,
                  COALESCE(s.min_value, 0.0) AS mn,
                  COALESCE(s.max_value, 100.0) AS mx
                FROM dbo.sensors s JOIN dbo.devices d ON d.id = s.device_id
                WHERE s.id = :sid
            """), {"sid": sensor_id}).fetchone()
        except Exception as exc:
            logger.warning("sql_sensor_profile_query_failed", extra={"series_id": series_id, "error": str(exc)})
            return self._default(series_id)
        if not row:
            return self._default(series_id)
        return self._build(series_id, row)

    def _build(self, series_id: str, row: Any) -> SensorProfile:
        eq = EquipmentClass.from_device_type(str(row.dt or ""))
        d = EQUIPMENT_DEFAULTS.get(eq, EQUIPMENT_DEFAULTS[EquipmentClass.GENERIC])
        mn = float(getattr(row, "mn", d["operational_range"][0]))
        mx = float(getattr(row, "mx", d["operational_range"][1]))
        return SensorProfile(
            series_id=series_id, equipment_class=eq,
            operational_range=(mn, mx), setpoint_tolerance=float(d["setpoint_tolerance"]),
            noise_floor=float(d["noise_floor"]), maintenance_history_score=0.5,
            hampel_k=float(d["hampel_k"]), hampel_window=int(d["hampel_window"]),
        )

    def _default(self, series_id: str) -> SensorProfile:
        d = EQUIPMENT_DEFAULTS[EquipmentClass.GENERIC]
        return SensorProfile(
            series_id=series_id, equipment_class=EquipmentClass.GENERIC,
            operational_range=(float(d["operational_range"][0]), float(d["operational_range"][1])),
            setpoint_tolerance=float(d["setpoint_tolerance"]), noise_floor=float(d["noise_floor"]),
            maintenance_history_score=0.5, hampel_k=float(d["hampel_k"]),
            hampel_window=int(d["hampel_window"]),
        )
