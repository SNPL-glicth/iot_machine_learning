"""Repository de umbrales — solo I/O contra dbo.alert_thresholds.

Extraído de prediction_service.py.  Responsabilidad única:
leer y escribir umbrales y eventos de threshold en la BD.
No contiene reglas de negocio ni decisiones.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Connection

from ...domain.services.threshold_evaluator import (
    ThresholdDefinition,
    ThresholdViolation,
)

logger = logging.getLogger(__name__)


class ThresholdRepository:
    """Acceso a datos de umbrales y eventos en SQL Server.

    Responsabilidad ÚNICA: queries SQL.  No evalúa ni decide.

    Attributes:
        _conn: Conexión sqlalchemy activa.
    """

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def load_active_threshold(
        self, sensor_id: int
    ) -> Optional[ThresholdDefinition]:
        """Carga el primer umbral activo de un sensor.

        Returns:
            ``ThresholdDefinition`` o ``None`` si no hay umbrales.
        """
        row = self._conn.execute(
            text(
                """
                SELECT TOP 1
                  id, condition_type, threshold_value_min,
                  threshold_value_max, severity, name
                FROM dbo.alert_thresholds
                WHERE sensor_id = :sensor_id AND is_active = 1
                ORDER BY id ASC
                """
            ),
            {"sensor_id": sensor_id},
        ).fetchone()

        if not row:
            return None

        return ThresholdDefinition(
            threshold_id=int(row[0]),
            condition_type=str(row[1]),
            value_min=float(row[2]) if row[2] is not None else None,
            value_max=float(row[3]) if row[3] is not None else None,
            severity=str(row[4]),
            name=str(row[5]),
        )

    def load_warning_range(
        self, sensor_id: int
    ) -> tuple[Optional[float], Optional[float]]:
        """Carga el rango WARNING del usuario para un sensor.

        Returns:
            Tupla ``(min, max)`` o ``(None, None)`` si no hay rango.
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
            return None, None

        return (
            float(row[0]) if row[0] is not None else None,
            float(row[1]) if row[1] is not None else None,
        )

    def has_recent_event(
        self,
        sensor_id: int,
        event_code: str,
        dedupe_minutes: int,
    ) -> bool:
        """Verifica si ya existe un evento reciente (deduplicación).

        Returns:
            ``True`` si hay un evento activo en los últimos ``dedupe_minutes``.
        """
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
            {
                "sensor_id": sensor_id,
                "event_code": event_code,
                "mins": dedupe_minutes,
            },
        ).fetchone()

        return row is not None

    def insert_threshold_event(
        self,
        *,
        sensor_id: int,
        device_id: int,
        prediction_id: int,
        violation: ThresholdViolation,
    ) -> None:
        """Inserta un evento de violación de umbral en dbo.ml_events."""
        self._conn.execute(
            text(
                """
                INSERT INTO dbo.ml_events (
                  device_id, sensor_id, prediction_id,
                  event_type, event_code, title, message,
                  status, created_at, payload
                )
                VALUES (
                  :device_id, :sensor_id, :prediction_id,
                  :event_type, :event_code, :title, :message,
                  'active', GETDATE(), :payload
                )
                """
            ),
            {
                "device_id": device_id,
                "sensor_id": sensor_id,
                "prediction_id": prediction_id,
                "event_type": violation.event_type,
                "event_code": "PRED_THRESHOLD_BREACH",
                "title": violation.title,
                "message": violation.message,
                "payload": json.dumps(violation.payload),
            },
        )

        logger.debug(
            "threshold_event_inserted",
            extra={
                "sensor_id": sensor_id,
                "threshold_id": violation.threshold_id,
                "event_type": violation.event_type,
            },
        )
