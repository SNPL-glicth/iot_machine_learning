"""Event persister service for ML online processing.

Extraído de ml_stream_runner.py para modularidad.
Responsabilidad: Persistir eventos ML y notificaciones en BD.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Optional

from sqlalchemy import text
from sqlalchemy.engine import Connection

from iot_ingest_services.common.db import get_engine
from iot_machine_learning.ml_service.repository.sensor_repository import get_device_id_for_sensor
from ..models.online_analysis import OnlineAnalysis

logger = logging.getLogger(__name__)


class MLEventPersister:
    """Persiste eventos ML y notificaciones en la base de datos.
    
    Responsabilidades:
    - Verificar cooldown de eventos
    - Verificar deduplicación
    - Insertar eventos ML
    - Crear notificaciones asociadas
    
    LÓGICA DE DOMINIO (FIX 2026-01-28 - Auditoría Delta Spike):
    - TODOS los eventos ML tienen cooldown de 5 min (incluyendo DELTA_SPIKE)
    - ALERT activo NO bloquea eventos ML (pueden coexistir)
    - Deduplicación para eventos activos del mismo tipo
    """
    
    def __init__(self) -> None:
        self._device_cache: Dict[int, int] = {}
    
    def _get_device_id(self, conn: Connection, sensor_id: int) -> int:
        """Obtiene device_id con cache."""
        if sensor_id in self._device_cache:
            return self._device_cache[sensor_id]
        device_id = get_device_id_for_sensor(conn, sensor_id)
        self._device_cache[sensor_id] = device_id
        return device_id

    def insert_ml_event(
        self,
        *,
        sensor_id: int,
        sensor_type: str,
        severity_label: str,
        event_type: str,
        event_code: str,
        title: str,
        explanation: str,
        recommended_action: str,
        analysis: OnlineAnalysis,
        ts_utc: float,
        prediction_id: Optional[int],
        extra_payload: Optional[dict],
    ) -> None:
        """Inserta un ml_event y crea una notificación asociada.

        - ml_events: registro detallado del evento ML.
        - alert_notifications: estado de notificación (read/unread).
        
        LÓGICA DE DOMINIO (FIX 2026-01-28 - Auditoría Delta Spike):
        - TODOS los eventos ML tienen cooldown de 5 min (incluyendo DELTA_SPIKE)
        - ALERT activo NO bloquea eventos ML (pueden coexistir)
        - Deduplicación para eventos activos del mismo tipo
        """
        engine = get_engine()
        with engine.begin() as conn:
            # Verificar cooldown
            recent_event = conn.execute(
                text("""
                    SELECT TOP 1 1 FROM dbo.ml_events
                    WHERE sensor_id = :sensor_id
                      AND event_code = :event_code
                      AND created_at >= DATEADD(MINUTE, -5, GETDATE())
                """),
                {"sensor_id": sensor_id, "event_code": event_code}
            ).fetchone()
            
            if recent_event:
                logger.debug(
                    "[ML_COOLDOWN] sensor_id=%s event_code=%s in cooldown, skipping",
                    sensor_id, event_code
                )
                return
            
            # Verificar deduplicación
            active_same_event = conn.execute(
                text("""
                    SELECT TOP 1 1 FROM dbo.ml_events
                    WHERE sensor_id = :sensor_id
                      AND event_code = :event_code
                      AND status = 'active'
                """),
                {"sensor_id": sensor_id, "event_code": event_code}
            ).fetchone()
            
            if active_same_event:
                logger.debug(
                    "[ML_DEDUPE] sensor_id=%s already has active %s event, skipping",
                    sensor_id, event_code
                )
                return

            device_id = self._get_device_id(conn, sensor_id)

            base_payload: dict = {
                "severity": severity_label,
                "behavior_pattern": analysis.behavior_pattern,
                "recommended_action": recommended_action,
                "sensor_type": sensor_type,
                "baseline_mean": analysis.baseline_mean,
                "last_value": analysis.last_value,
                "z_score_last": analysis.z_score_last,
                "is_curve_anomalous": analysis.is_curve_anomalous,
                "has_microvariation": analysis.has_microvariation,
                "microvariation_delta": analysis.microvariation_delta,
            }
            if extra_payload:
                base_payload.update(extra_payload)

            payload_json = json.dumps(base_payload, ensure_ascii=False)

            # Insertar evento ML
            row = conn.execute(
                text(
                    """
                    INSERT INTO dbo.ml_events (
                      device_id,
                      sensor_id,
                      prediction_id,
                      event_type,
                      event_code,
                      title,
                      message,
                      status,
                      created_at,
                      payload
                    )
                    OUTPUT INSERTED.id
                    VALUES (
                      :device_id,
                      :sensor_id,
                      :prediction_id,
                      :event_type,
                      :event_code,
                      :title,
                      :message,
                      'active',
                      DATEADD(second, :ts_utc, '1970-01-01'),
                      :payload
                    )
                    """
                ),
                {
                    "device_id": device_id,
                    "sensor_id": sensor_id,
                    "prediction_id": prediction_id,
                    "event_type": event_type,
                    "event_code": event_code,
                    "title": title,
                    "message": explanation,
                    "ts_utc": ts_utc,
                    "payload": payload_json,
                },
            ).fetchone()

            if not row:
                return

            event_id = int(row[0])

            # Crear notificación
            try:
                conn.execute(
                    text(
                        """
                        INSERT INTO dbo.alert_notifications (
                          source,
                          source_event_id,
                          severity,
                          title,
                          message,
                          is_read,
                          created_at
                        )
                        VALUES (
                          :source,
                          :source_event_id,
                          :severity,
                          :title,
                          :message,
                          0,
                          GETDATE()
                        )
                        """
                    ),
                    {
                        "source": "ml_event",
                        "source_event_id": event_id,
                        "severity": event_type,
                        "title": title,
                        "message": explanation,
                    },
                )
            except Exception:
                logger.exception("No se pudo insertar en alert_notifications (ML online)")

    def should_dedupe_prediction_deviation(
        self,
        conn: Connection,
        *,
        sensor_id: int,
        dedupe_minutes: int,
    ) -> bool:
        """Verifica si hay evento PREDICTION_DEVIATION reciente."""
        row = conn.execute(
            text(
                """
                SELECT TOP 1 1
                FROM dbo.ml_events
                WHERE sensor_id = :sensor_id
                  AND event_code = 'PREDICTION_DEVIATION'
                  AND status IN ('active', 'acknowledged')
                  AND created_at >= DATEADD(minute, -:mins, GETDATE())
                ORDER BY created_at DESC
                """
            ),
            {"sensor_id": sensor_id, "mins": dedupe_minutes},
        ).fetchone()
        return row is not None
