"""Base queries module - Sensor readings and metadata."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List

from sqlalchemy import text
from sqlalchemy.engine import Connection

from .....domain.entities.sensor_reading import SensorReading, SensorWindow

logger = logging.getLogger(__name__)


def _safe_float(value: object, default: float = 0.0) -> float:
    """Convierte a float con guard contra None/NaN/Inf."""
    if value is None:
        return default
    try:
        import math
        f = float(value)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _is_valid_sensor_value(value: object) -> bool:
    """True si el valor es un número finito no-None."""
    if value is None:
        return False
    try:
        import math
        f = float(value)
        return math.isfinite(f)
    except (TypeError, ValueError):
        return False


class BaseQueries:
    """Queries base para lecturas de sensores y metadata."""
    
    def __init__(self, conn: Connection) -> None:
        self._conn = conn
    
    def load_sensor_window(
        self,
        sensor_id: int,
        limit: int = 500,
    ) -> SensorWindow:
        """Carga las últimas ``limit`` lecturas de un sensor."""
        rows = self._conn.execute(
            text(
                """
                SELECT TOP (:limit) [value], [timestamp]
                FROM dbo.sensor_readings
                WHERE sensor_id = :sensor_id
                ORDER BY [timestamp] DESC
                """
            ),
            {"sensor_id": sensor_id, "limit": limit},
        ).fetchall()

        rows = list(reversed(rows))

        readings: List[SensorReading] = []
        for row in rows:
            raw_value = row[0]
            raw_ts = row[1]

            if not _is_valid_sensor_value(raw_value):
                continue

            value = _safe_float(raw_value)

            if isinstance(raw_ts, datetime):
                ts = raw_ts.replace(tzinfo=timezone.utc).timestamp()
            elif raw_ts is not None:
                ts = float(raw_ts)
            else:
                ts = 0.0

            readings.append(SensorReading(
                sensor_id=sensor_id,
                value=value,
                timestamp=ts,
            ))

        return SensorWindow(sensor_id=sensor_id, readings=readings)

    def list_active_sensor_ids(self) -> List[int]:
        """Retorna IDs de sensores activos."""
        rows = self._conn.execute(
            text("SELECT id FROM dbo.sensors WHERE is_active = 1")
        ).fetchall()
        return [int(r[0]) for r in rows]
    
    def get_sensor_metadata(self, sensor_id: int) -> Dict[str, object]:
        """Obtiene metadata de un sensor."""
        row = self._conn.execute(
            text(
                """
                SELECT id, device_id, sensor_type, name, unit, location
                FROM dbo.sensors
                WHERE id = :sensor_id
                """
            ),
            {"sensor_id": sensor_id},
        ).fetchone()

        if not row:
            return {"sensor_id": sensor_id}

        return {
            "sensor_id": int(row[0]),
            "device_id": int(row[1]) if row[1] else 0,
            "sensor_type": str(row[2] or ""),
            "name": str(row[3] or ""),
            "unit": str(row[4] or ""),
            "location": str(row[5] or ""),
        }

    def get_device_id_for_sensor(self, sensor_id: int) -> int:
        """Obtiene el device_id de un sensor."""
        row = self._conn.execute(
            text("SELECT device_id FROM dbo.sensors WHERE id = :sensor_id"),
            {"sensor_id": sensor_id},
        ).fetchone()

        if not row:
            raise ValueError(f"Sensor {sensor_id} not found")

        return int(row[0])
