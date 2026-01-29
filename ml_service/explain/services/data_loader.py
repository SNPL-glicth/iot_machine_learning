"""Data loader service for contextual explainer.

Extracts data access logic from ContextualExplainer.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)


class ExplainerDataLoader:
    """Loads data from database for contextual explanations."""
    
    def __init__(self, conn: Connection):
        self._conn = conn
    
    def get_sensor_info(self, sensor_id: int) -> dict:
        """Obtiene información del sensor y su dispositivo."""
        try:
            row = self._conn.execute(
                text("""
                    SELECT 
                        s.id, s.sensor_type, s.name AS sensor_name, s.unit,
                        d.id AS device_id, d.name AS device_name, d.device_type
                    FROM dbo.sensors s
                    JOIN dbo.devices d ON d.id = s.device_id
                    WHERE s.id = :sensor_id
                """),
                {"sensor_id": sensor_id},
            ).fetchone()
            
            if row:
                return {
                    "sensor_type": str(row.sensor_type or "unknown"),
                    "sensor_name": str(row.sensor_name or f"Sensor {sensor_id}"),
                    "unit": str(row.unit or ""),
                    "device_id": int(row.device_id),
                    "device_name": str(row.device_name or "Dispositivo"),
                    "device_type": str(row.device_type or ""),
                }
        except Exception as e:
            logger.warning("Failed to get sensor info: %s", str(e))
        
        return {}
    
    def get_current_value(self, sensor_id: int) -> float:
        """Obtiene el valor actual del sensor."""
        try:
            row = self._conn.execute(
                text("""
                    SELECT TOP 1 value
                    FROM dbo.sensor_readings
                    WHERE sensor_id = :sensor_id
                    ORDER BY timestamp DESC
                """),
                {"sensor_id": sensor_id},
            ).fetchone()
            
            if row and row[0] is not None:
                return float(row[0])
        except Exception as e:
            logger.warning("Failed to get current value: %s", str(e))
        
        return 0.0
    
    def get_user_thresholds(self, sensor_id: int) -> dict:
        """Obtiene los umbrales definidos por el usuario."""
        try:
            row = self._conn.execute(
                text("""
                    SELECT threshold_value_min, threshold_value_max
                    FROM dbo.alert_thresholds
                    WHERE sensor_id = :sensor_id
                      AND is_active = 1
                      AND condition_type = 'out_of_range'
                    ORDER BY id ASC
                """),
                {"sensor_id": sensor_id},
            ).fetchone()
            
            if row:
                return {
                    "min": float(row[0]) if row[0] is not None else None,
                    "max": float(row[1]) if row[1] is not None else None,
                }
        except Exception as e:
            logger.warning("Failed to get user thresholds: %s", str(e))
        
        return {}
    
    def get_recent_stats(self, sensor_id: int, hours: int = 24) -> dict:
        """Obtiene estadísticas de las últimas horas."""
        try:
            row = self._conn.execute(
                text("""
                    SELECT 
                        AVG(value) AS avg_val,
                        MIN(value) AS min_val,
                        MAX(value) AS max_val,
                        STDEV(value) AS std_val
                    FROM dbo.sensor_readings
                    WHERE sensor_id = :sensor_id
                      AND timestamp >= DATEADD(hour, -:hours, GETDATE())
                """),
                {"sensor_id": sensor_id, "hours": hours},
            ).fetchone()
            
            if row and row.avg_val is not None:
                return {
                    "avg": float(row.avg_val),
                    "min": float(row.min_val) if row.min_val else None,
                    "max": float(row.max_val) if row.max_val else None,
                    "std": float(row.std_val) if row.std_val else None,
                }
        except Exception as e:
            logger.warning("Failed to get recent stats: %s", str(e))
        
        return {}
    
    def get_correlated_events(self, sensor_id: int, minutes: int = 30) -> list[dict]:
        """Obtiene eventos de sensores correlacionados (mismo dispositivo)."""
        try:
            rows = self._conn.execute(
                text("""
                    SELECT 
                        e.id, e.sensor_id, s.sensor_type, e.event_type, e.title
                    FROM dbo.ml_events e
                    JOIN dbo.sensors s ON s.id = e.sensor_id
                    WHERE s.device_id = (SELECT device_id FROM dbo.sensors WHERE id = :sensor_id)
                      AND e.sensor_id != :sensor_id
                      AND e.created_at >= DATEADD(minute, -:minutes, GETDATE())
                      AND e.status IN ('active', 'acknowledged')
                """),
                {"sensor_id": sensor_id, "minutes": minutes},
            ).fetchall()
            
            return [
                {
                    "event_id": int(r.id),
                    "sensor_id": int(r.sensor_id),
                    "sensor_type": str(r.sensor_type or ""),
                    "event_type": str(r.event_type or ""),
                    "title": str(r.title or ""),
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning("Failed to get correlated events: %s", str(e))
        
        return []
    
    def get_similar_events_history(self, sensor_id: int, is_anomaly: bool) -> dict:
        """Obtiene historial de eventos similares."""
        try:
            event_code = "ANOMALY_DETECTED" if is_anomaly else "PRED_THRESHOLD_BREACH"
            
            row = self._conn.execute(
                text("""
                    SELECT 
                        COUNT(*) AS cnt,
                        MAX(created_at) AS last_at
                    FROM dbo.ml_events
                    WHERE sensor_id = :sensor_id
                      AND event_code = :event_code
                      AND created_at >= DATEADD(day, -30, GETDATE())
                """),
                {"sensor_id": sensor_id, "event_code": event_code},
            ).fetchone()
            
            if row and row.cnt > 0:
                return {
                    "count": int(row.cnt),
                    "last_at": row.last_at,
                }
        except Exception as e:
            logger.warning("Failed to get similar events history: %s", str(e))
        
        return {"count": 0, "last_at": None}
