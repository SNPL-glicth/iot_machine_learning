"""Model service for ML API.

Handles ML model management operations.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)


class ModelService:
    """Service for managing ML models.
    
    Responsibilities:
    - Get active model for sensor
    - Create new models
    - Activate/deactivate models
    """
    
    def __init__(self, conn: Connection):
        self._conn = conn
    
    def get_active_model(self, sensor_id: int) -> Optional[dict]:
        """Get the active model for a sensor."""
        row = self._conn.execute(
            text(
                """
                SELECT id, sensor_id, model_name, model_type, version, is_active, trained_at
                FROM dbo.ml_models
                WHERE sensor_id = :sensor_id AND is_active = 1
                ORDER BY trained_at DESC
                """
            ),
            {"sensor_id": sensor_id},
        ).fetchone()
        
        if not row:
            return None
        
        return {
            "id": row[0],
            "sensor_id": row[1],
            "model_name": row[2],
            "model_type": row[3],
            "version": row[4],
            "is_active": row[5],
            "trained_at": row[6],
        }
    
    def list_models(self, sensor_id: Optional[int] = None) -> list[dict]:
        """List all models, optionally filtered by sensor."""
        if sensor_id:
            rows = self._conn.execute(
                text(
                    """
                    SELECT id, sensor_id, model_name, model_type, version, is_active, trained_at
                    FROM dbo.ml_models
                    WHERE sensor_id = :sensor_id
                    ORDER BY trained_at DESC
                    """
                ),
                {"sensor_id": sensor_id},
            ).fetchall()
        else:
            rows = self._conn.execute(
                text(
                    """
                    SELECT TOP 100 id, sensor_id, model_name, model_type, version, is_active, trained_at
                    FROM dbo.ml_models
                    ORDER BY trained_at DESC
                    """
                ),
            ).fetchall()
        
        return [
            {
                "id": row[0],
                "sensor_id": row[1],
                "model_name": row[2],
                "model_type": row[3],
                "version": row[4],
                "is_active": row[5],
                "trained_at": row[6],
            }
            for row in rows
        ]
