"""Queries de base de datos para correlación de sensores."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.engine import Connection

from .types import DeviceSensorGroup, SensorSnapshot

logger = logging.getLogger(__name__)


def get_device_sensors(conn: Connection, device_id: int) -> DeviceSensorGroup:
    """Obtiene todos los sensores activos de un dispositivo con sus últimas predicciones.
    
    Args:
        conn: SQLAlchemy connection
        device_id: ID del dispositivo
        
    Returns:
        DeviceSensorGroup con todos los sensores del dispositivo
    """
    rows = conn.execute(
        text("""
            SELECT 
                s.id AS sensor_id,
                s.sensor_type,
                d.id AS device_id,
                d.name AS device_name,
                p.predicted_value,
                p.trend,
                p.anomaly_score,
                p.severity,
                p.predicted_at,
                (SELECT TOP 1 sr.value 
                 FROM dbo.sensor_readings sr 
                 WHERE sr.sensor_id = s.id 
                 ORDER BY sr.timestamp DESC) AS current_value
            FROM dbo.sensors s
            JOIN dbo.devices d ON d.id = s.device_id
            LEFT JOIN dbo.predictions p ON p.sensor_id = s.id
                AND p.id = (
                    SELECT TOP 1 p2.id 
                    FROM dbo.predictions p2 
                    WHERE p2.sensor_id = s.id 
                    ORDER BY p2.predicted_at DESC
                )
            WHERE s.device_id = :device_id
              AND s.is_active = 1
        """),
        {"device_id": device_id},
    ).fetchall()
    
    device_name = ""
    sensors = []
    
    for row in rows:
        device_name = row.device_name or f"Device {device_id}"
        
        if row.predicted_value is not None:
            sensors.append(SensorSnapshot(
                sensor_id=int(row.sensor_id),
                sensor_type=str(row.sensor_type or "unknown").lower(),
                current_value=float(row.current_value) if row.current_value else 0.0,
                predicted_value=float(row.predicted_value),
                trend=str(row.trend or "stable").lower(),
                anomaly_score=float(row.anomaly_score) if row.anomaly_score else 0.0,
                severity=str(row.severity or "info").lower(),
                timestamp=row.predicted_at or datetime.now(timezone.utc),
            ))
    
    return DeviceSensorGroup(
        device_id=device_id,
        device_name=device_name,
        sensors=sensors,
    )


def get_device_id_for_sensor(conn: Connection, sensor_id: int) -> int | None:
    """Obtiene el device_id para un sensor dado.
    
    Args:
        conn: SQLAlchemy connection
        sensor_id: ID del sensor
        
    Returns:
        device_id o None si no se encuentra
    """
    row = conn.execute(
        text("SELECT device_id FROM dbo.sensors WHERE id = :sensor_id"),
        {"sensor_id": sensor_id},
    ).fetchone()
    
    if not row:
        return None
    
    return int(row[0])


def get_correlated_events(
    conn: Connection,
    sensor_id: int,
    time_window_minutes: int = 30,
) -> list[dict]:
    """Obtiene eventos ML recientes de sensores correlacionados (mismo dispositivo).
    
    Args:
        conn: SQLAlchemy connection
        sensor_id: ID del sensor de referencia
        time_window_minutes: Ventana de tiempo en minutos
        
    Returns:
        Lista de eventos correlacionados
    """
    rows = conn.execute(
        text("""
            SELECT 
                e.id AS event_id,
                e.sensor_id,
                s.sensor_type,
                e.event_type,
                e.event_code,
                e.title,
                e.created_at
            FROM dbo.ml_events e
            JOIN dbo.sensors s ON s.id = e.sensor_id
            WHERE s.device_id = (
                SELECT device_id FROM dbo.sensors WHERE id = :sensor_id
            )
            AND e.sensor_id != :sensor_id
            AND e.created_at >= DATEADD(minute, -:minutes, GETDATE())
            AND e.status IN ('active', 'acknowledged')
            ORDER BY e.created_at DESC
        """),
        {"sensor_id": sensor_id, "minutes": time_window_minutes},
    ).fetchall()
    
    return [
        {
            "event_id": int(r.event_id),
            "sensor_id": int(r.sensor_id),
            "sensor_type": str(r.sensor_type or "unknown"),
            "event_type": str(r.event_type or "notice"),
            "event_code": str(r.event_code or ""),
            "title": str(r.title or ""),
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]
