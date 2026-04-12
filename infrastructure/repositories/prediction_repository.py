"""Repositorio para queries de predicciones.

Responsabilidad única: Queries SQL relacionadas con predicciones.
Separado de StoragePort (que maneja persistencia).
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)


class PredictionRepository:
    """Repositorio para queries de predicciones.
    
    Responsabilidades:
    - Obtener ID de última predicción insertada
    - Queries de lectura relacionadas con predicciones
    
    NO incluye:
    - Persistencia (eso es StoragePort)
    - Lógica de negocio (eso es domain)
    """
    
    def __init__(self, conn: Connection):
        """Inicializa con conexión SQL.
        
        Args:
            conn: Conexión SQLAlchemy
        """
        self._conn = conn
    
    def get_latest_prediction_id(self, sensor_id: int) -> int:
        """Obtiene el ID de la última predicción insertada.
        
        Args:
            sensor_id: ID del sensor
        
        Returns:
            ID de la predicción, o 0 si no existe
        """
        try:
            row = self._conn.execute(
                text(
                    """
                    SELECT TOP 1 id
                    FROM dbo.predictions
                    WHERE sensor_id = :sensor_id
                    ORDER BY predicted_at DESC
                    """
                ),
                {"sensor_id": sensor_id},
            ).fetchone()
            
            return int(row[0]) if row else 0
        
        except Exception as exc:
            logger.warning(
                "get_latest_prediction_id_failed",
                extra={"sensor_id": sensor_id, "error": str(exc)},
            )
            return 0
    
    def get_prediction_by_id(self, prediction_id: int) -> Optional[dict]:
        """Obtiene una predicción por ID.
        
        Args:
            prediction_id: ID de la predicción
        
        Returns:
            Dict con datos de la predicción, o None si no existe
        """
        try:
            row = self._conn.execute(
                text(
                    """
                    SELECT 
                        id, sensor_id, model_id, predicted_value,
                        confidence, predicted_at, target_timestamp,
                        horizon_minutes, window_points, trend,
                        engine_name, audit_trace_id
                    FROM dbo.predictions
                    WHERE id = :prediction_id
                    """
                ),
                {"prediction_id": prediction_id},
            ).fetchone()
            
            if not row:
                return None
            
            return {
                "id": row[0],
                "sensor_id": row[1],
                "model_id": row[2],
                "predicted_value": float(row[3]),
                "confidence": float(row[4]) if row[4] else None,
                "predicted_at": row[5],
                "target_timestamp": row[6],
                "horizon_minutes": row[7],
                "window_points": row[8],
                "trend": row[9],
                "engine_name": row[10],
                "audit_trace_id": row[11],
            }
        
        except Exception as exc:
            logger.warning(
                "get_prediction_by_id_failed",
                extra={"prediction_id": prediction_id, "error": str(exc)},
            )
            return None
