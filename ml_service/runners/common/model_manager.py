"""Gestión de modelos ML.

Responsabilidad única: Crear y obtener IDs de modelos ML en BD.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)


class ModelManager:
    """Gestiona modelos ML en la base de datos.
    
    Responsabilidades:
    - Obtener o crear modelo para un sensor
    - Gestionar versiones de modelos (dinámico o configurado)
    """
    
    DEFAULT_MODEL_NAME = "sklearn_regression_iforest"
    DEFAULT_MODEL_TYPE = "sklearn"
    
    def __init__(self, version: Optional[str] = None):
        """Initialize ModelManager with optional version.
        
        Args:
            version: Optional version string. If not provided, uses
                    auto-generated version based on timestamp.
        """
        self._version = version
    
    def _get_version(self) -> str:
        """Get version string - either configured or auto-generated."""
        if self._version is not None:
            return self._version
        # Auto-generate version based on current timestamp
        return datetime.now(timezone.utc).strftime("%Y.%m.%d.%H%M%S")
    
    def get_or_create_model_id(self, conn: Connection, sensor_id: int) -> int:
        """Obtiene o crea el ID del modelo para un sensor.
        
        Args:
            conn: Conexión a BD
            sensor_id: ID del sensor
            
        Returns:
            ID del modelo
        """
        # Buscar modelo existente
        row = conn.execute(
            text(
                """
                SELECT TOP 1 id
                FROM dbo.ml_models
                WHERE sensor_id = :sensor_id AND is_active = 1 AND model_type = 'sklearn'
                ORDER BY trained_at DESC
                """
            ),
            {"sensor_id": sensor_id},
        ).fetchone()

        if row:
            return int(row[0])

        # Crear nuevo modelo
        created = conn.execute(
            text(
                """
                INSERT INTO dbo.ml_models (
                  sensor_id, model_name, model_type, version, is_active, trained_at
                )
                OUTPUT INSERTED.id
                VALUES (
                  :sensor_id, :model_name, :model_type, :version, 1, GETDATE()
                )
                """
            ),
            {
                "sensor_id": sensor_id,
                "model_name": self.DEFAULT_MODEL_NAME,
                "model_type": self.DEFAULT_MODEL_TYPE,
                "version": self._get_version(),
            },
        ).fetchone()

        if not created:
            raise RuntimeError(f"Failed to create ml_models row for sensor {sensor_id}")

        model_id = int(created[0])
        logger.debug("[MODEL_MANAGER] Created model id=%s for sensor=%s", model_id, sensor_id)
        return model_id
