"""Threshold service for ML API.

Handles threshold validation and event deduplication.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)


class ThresholdService:
    """Service for threshold validation.
    
    Responsibilities:
    - Check if value is within warning range
    - Check event deduplication
    - Get threshold configuration
    """
    
    def __init__(self, conn: Connection):
        self._conn = conn
    
    def is_value_within_warning_range(self, sensor_id: int, value: float) -> bool:
        """Check if value is within user-defined warning range.
        
        DOMAIN RULE:
        If value is within [warning_min, warning_max], ML should NOT
        generate events. User defined this range as "normal".
        """
        row = self._conn.execute(
            text(
                """
                SELECT threshold_value_min, threshold_value_max
                FROM dbo.alert_thresholds
                WHERE sensor_id = :sensor_id
                  AND is_active = 1
                  AND severity = 'warning'
                  AND condition_type = 'out_of_range'
                ORDER BY id ASC
                """
            ),
            {"sensor_id": sensor_id},
        ).fetchone()
        
        if not row:
            return False
        
        warning_min = float(row[0]) if row[0] is not None else None
        warning_max = float(row[1]) if row[1] is not None else None
        
        if warning_min is None and warning_max is None:
            return False
        
        if warning_min is not None and value < warning_min:
            return False
        if warning_max is not None and value > warning_max:
            return False
        
        return True
    
    def should_dedupe_event(
        self,
        sensor_id: int,
        event_code: str,
        dedupe_minutes: int,
    ) -> bool:
        """Check if event should be deduplicated."""
        row = self._conn.execute(
            text(
                """
                SELECT TOP 1 1
                FROM dbo.ml_events
                WHERE sensor_id = :sensor_id
                  AND event_code = :event_code
                  AND status IN ('active', 'acknowledged')
                  AND created_at >= DATEADD(minute, -:mins, GETDATE())
                ORDER BY created_at DESC
                """
            ),
            {"sensor_id": sensor_id, "event_code": event_code, "mins": dedupe_minutes},
        ).fetchone()
        
        return row is not None
    
    def get_thresholds(self, sensor_id: int) -> list[dict]:
        """Get all thresholds for a sensor."""
        rows = self._conn.execute(
            text(
                """
                SELECT id, condition_type, threshold_value_min, threshold_value_max, 
                       severity, name, is_active
                FROM dbo.alert_thresholds
                WHERE sensor_id = :sensor_id
                ORDER BY id ASC
                """
            ),
            {"sensor_id": sensor_id},
        ).fetchall()
        
        return [
            {
                "id": row[0],
                "condition_type": row[1],
                "threshold_value_min": float(row[2]) if row[2] is not None else None,
                "threshold_value_max": float(row[3]) if row[3] is not None else None,
                "severity": row[4],
                "name": row[5],
                "is_active": row[6],
            }
            for row in rows
        ]
