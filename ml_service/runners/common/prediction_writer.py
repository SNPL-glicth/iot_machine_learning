"""Escritor de predicciones — LEGACY.

Responsabilidad única: Persistir predicciones en la base de datos.

.. deprecated::
    Used only by the legacy dev/test batch runner
    (``ml_service.runners.ml_batch_runner``).
    The official production writer is:
        ``infrastructure.adapters.sqlserver_storage.SqlServerStorageAdapter.save_prediction()``
    Both writers now write the same 16 columns. This module will be
    consolidated into the enterprise path in a future cleanup.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.engine import Connection

if TYPE_CHECKING:
    from iot_machine_learning.ml_service.explain.explanation_builder import PredictionExplanation

logger = logging.getLogger(__name__)


class PredictionWriter:
    """Persiste predicciones en la base de datos.
    
    Responsabilidades:
    - Insertar predicciones
    - Ajustar severidad si hay delta spike reciente
    """
    
    def __init__(self, event_writer: "EventWriter"):
        self._event_writer = event_writer
    
    def insert_prediction(
        self,
        conn: Connection,
        *,
        model_id: int,
        sensor_id: int,
        device_id: int,
        explanation: "PredictionExplanation",
        target_ts_utc: datetime,
        horizon_minutes: int,
        window_points: int,
    ) -> int:
        """Inserta una predicción en la base de datos.
        
        Args:
            conn: Conexión a BD
            model_id: ID del modelo ML
            sensor_id: ID del sensor
            device_id: ID del dispositivo
            explanation: Explicación de la predicción
            target_ts_utc: Timestamp objetivo (UTC)
            horizon_minutes: Horizonte de predicción
            window_points: Puntos de ventana usados
            
        Returns:
            ID de la predicción insertada
        """
        row = conn.execute(
            text(
                """
                INSERT INTO dbo.predictions (
                  model_id,
                  sensor_id,
                  device_id,
                  predicted_value,
                  confidence,
                  predicted_at,
                  target_timestamp,
                  horizon_minutes,
                  window_points,
                  trend,
                  is_anomaly,
                  anomaly_score,
                  risk_level,
                  severity,
                  explanation,
                  status
                )
                OUTPUT INSERTED.id
                VALUES (
                  :model_id,
                  :sensor_id,
                  :device_id,
                  :predicted_value,
                  :confidence,
                  GETDATE(),
                  :target_timestamp,
                  :horizon_minutes,
                  :window_points,
                  :trend,
                  :is_anomaly,
                  :anomaly_score,
                  :risk_level,
                  :severity,
                  :explanation,
                  'active'
                )
                """
            ),
            {
                "model_id": model_id,
                "sensor_id": sensor_id,
                "device_id": device_id,
                "predicted_value": explanation.predicted_value,
                "confidence": explanation.confidence,
                "target_timestamp": target_ts_utc.replace(tzinfo=None),
                "horizon_minutes": horizon_minutes,
                "window_points": window_points,
                "trend": explanation.trend,
                "is_anomaly": 1 if explanation.anomaly else 0,
                "anomaly_score": explanation.anomaly_score,
                "risk_level": explanation.risk_level,
                "severity": explanation.severity,
                "explanation": explanation.explanation,
            },
        ).fetchone()

        if not row:
            raise RuntimeError(f"Failed to insert prediction for sensor {sensor_id}")

        prediction_id = int(row[0])
        
        # Ajustar severidad si hay delta spike reciente
        self._adjust_for_delta_spike(conn, sensor_id, prediction_id)
        
        logger.debug(
            "[PREDICTION_WRITER] Inserted id=%s sensor=%s pred=%.4f",
            prediction_id, sensor_id, explanation.predicted_value
        )
        
        return prediction_id
    
    def _adjust_for_delta_spike(
        self, 
        conn: Connection, 
        sensor_id: int, 
        prediction_id: int
    ) -> None:
        """Ajusta severidad si hay delta spike reciente."""
        if not self._event_writer.has_recent_delta_spike(conn, sensor_id, window_seconds=30):
            return
        
        conn.execute(
            text(
                """
                UPDATE dbo.predictions
                SET
                  anomaly_score = CASE
                    WHEN anomaly_score IS NULL OR anomaly_score < 0.3 THEN 0.3
                    ELSE anomaly_score
                  END,
                  severity = CASE
                    WHEN UPPER(severity) = 'CRITICAL' THEN severity
                    WHEN severity IS NULL OR UPPER(severity) NOT IN ('WARNING','CRITICAL') THEN 'WARNING'
                    ELSE severity
                  END
                WHERE id = :id
                """
            ),
            {"id": prediction_id},
        )
        logger.debug(
            "[PREDICTION_WRITER] Adjusted severity for delta spike sensor=%s pred=%s",
            sensor_id, prediction_id
        )


# Import para type hints (al final para evitar circular imports)
try:
    from .event_writer import EventWriter
except ImportError:
    from event_writer import EventWriter
