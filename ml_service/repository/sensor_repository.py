from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from sqlalchemy import text
from sqlalchemy.engine import Connection


@dataclass(frozen=True)
class SensorSeries:
    sensor_id: int
    timestamps: list[datetime]
    values: list[float]


@dataclass(frozen=True)
class SensorMetadata:
    sensor_id: int
    sensor_type: str
    unit: str
    location: str
    criticality: str  # low | medium | high


def list_active_sensors(conn: Connection) -> Iterable[int]:
    rows = conn.execute(
        text("SELECT id FROM dbo.sensors WHERE is_active = 1 ORDER BY id ASC")
    ).fetchall()
    for r in rows:
        yield int(r[0])


def load_sensor_series(
    conn: Connection,
    sensor_id: int,
    limit_points: int,
) -> SensorSeries:
    # FIX FASE2: Evitar CAST AS float, mantener precisión DECIMAL(15,5)
    rows = conn.execute(
        text(
            """
            SELECT TOP (:limit) [timestamp], [value] AS v
            FROM dbo.sensor_readings
            WHERE sensor_id = :sensor_id
            ORDER BY [timestamp] ASC
            """
        ),
        {"sensor_id": sensor_id, "limit": limit_points},
    ).fetchall()

    ts: list[datetime] = []
    vals: list[float] = []
    for t, v in rows:
        ts.append(t)
        # Python Decimal → float mantiene mejor precisión que SQL CAST
        vals.append(float(v) if v is not None else 0.0)

    return SensorSeries(sensor_id=sensor_id, timestamps=ts, values=vals)


def get_device_id_for_sensor(conn: Connection, sensor_id: int) -> int:
    row = conn.execute(
        text("SELECT device_id FROM dbo.sensors WHERE id = :sensor_id"),
        {"sensor_id": sensor_id},
    ).fetchone()
    if not row:
        raise RuntimeError(f"sensor_id not found: {sensor_id}")
    return int(row[0])


def load_sensor_metadata(conn: Connection, sensor_id: int) -> SensorMetadata:
    """Carga metadata básica del sensor + dispositivo para contextualizar ML.

    - sensor_type, unit: vienen de dbo.sensors
    - location: usamos el nombre del dispositivo como localización amigable
    - criticality: heurística por tipo de sensor (puede mejorarse más adelante)
    """

    row = conn.execute(
        text(
            """
            SELECT TOP 1
              s.id AS sensor_id,
              COALESCE(s.sensor_type, '') AS sensor_type,
              COALESCE(s.unit, '') AS unit,
              COALESCE(d.name, '') AS device_name,
              COALESCE(d.device_type, '') AS device_type
            FROM dbo.sensors s
            JOIN dbo.devices d ON d.id = s.device_id
            WHERE s.id = :sensor_id
            """
        ),
        {"sensor_id": sensor_id},
    ).fetchone()

    if not row:
        raise RuntimeError(f"sensor_id not found for metadata: {sensor_id}")

    sensor_type = str(row.sensor_type or '').lower().strip()
    unit = str(row.unit or '').strip()
    location = str(row.device_name or '').strip() or 'ubicación no especificada'

    # Heurística simple de criticidad por tipo de sensor.
    if sensor_type in {"temperature", "power", "voltage"}:
        criticality = "high"
    elif sensor_type in {"humidity", "air_quality"}:
        criticality = "medium"
    else:
        criticality = "medium"

    return SensorMetadata(
        sensor_id=int(row.sensor_id),
        sensor_type=sensor_type or 'unknown',
        unit=unit or '-',
        location=location,
        criticality=criticality,
    )
