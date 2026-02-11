"""Adapter: SQL Server como implementación de StoragePort.

Convierte las queries SQL Server existentes en la interfaz
de dominio StoragePort.  Extrae la lógica de I/O que antes
vivía dispersa en prediction_service.py y sensor_processor.py.

Patrón Adapter (GoF): convierte sqlalchemy.Connection a StoragePort
sin modificar el dominio ni la BD.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.engine import Connection

from ...domain.entities.anomaly import AnomalyResult
from ...domain.entities.prediction import Prediction
from ...domain.entities.sensor_reading import SensorReading, SensorWindow
from ...domain.ports.storage_port import StoragePort

logger = logging.getLogger(__name__)


def _safe_float(value: object, default: float = 0.0) -> float:
    """Convierte a float con guard contra None/NaN/Inf."""
    if value is None:
        return default
    try:
        import math
        f = float(value)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _is_valid_sensor_value(value: object) -> bool:
    """True si el valor es un número finito no-None."""
    if value is None:
        return False
    try:
        import math
        f = float(value)
        return math.isfinite(f)
    except (TypeError, ValueError):
        return False


class SqlServerStorageAdapter(StoragePort):
    """Implementación de StoragePort para SQL Server vía sqlalchemy.

    Responsabilidad ÚNICA: traducir operaciones de dominio a queries SQL.
    No contiene lógica de negocio, cálculos ni decisiones.

    Attributes:
        _conn: Conexión sqlalchemy activa.
    """

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    # --- Lecturas de sensores ---

    def load_sensor_window(
        self,
        sensor_id: int,
        limit: int = 500,
    ) -> SensorWindow:
        """Carga las últimas ``limit`` lecturas de un sensor.

        Query: dbo.sensor_readings ORDER BY timestamp DESC.
        Retorna en orden cronológico (más antiguo primero).
        """
        rows = self._conn.execute(
            text(
                """
                SELECT TOP (:limit) [value], [timestamp]
                FROM dbo.sensor_readings
                WHERE sensor_id = :sensor_id
                ORDER BY [timestamp] DESC
                """
            ),
            {"sensor_id": sensor_id, "limit": limit},
        ).fetchall()

        # Invertir para orden cronológico (más antiguo primero)
        rows = list(reversed(rows))

        readings: List[SensorReading] = []
        for row in rows:
            raw_value = row[0]
            raw_ts = row[1]

            if not _is_valid_sensor_value(raw_value):
                continue

            value = _safe_float(raw_value)

            # Convertir timestamp a float (epoch seconds)
            if isinstance(raw_ts, datetime):
                ts = raw_ts.replace(tzinfo=timezone.utc).timestamp()
            elif raw_ts is not None:
                ts = float(raw_ts)
            else:
                ts = 0.0

            readings.append(SensorReading(
                sensor_id=sensor_id,
                value=value,
                timestamp=ts,
            ))

        return SensorWindow(sensor_id=sensor_id, readings=readings)

    def list_active_sensor_ids(self) -> List[int]:
        """Retorna IDs de sensores activos."""
        rows = self._conn.execute(
            text("SELECT id FROM dbo.sensors WHERE is_active = 1")
        ).fetchall()
        return [int(r[0]) for r in rows]

    # --- Predicciones ---

    def save_prediction(self, prediction: Prediction) -> int:
        """Persiste una predicción en dbo.predictions.

        Crea el modelo en dbo.ml_models si no existe.
        """
        model_id = self._get_or_create_model_id(prediction.sensor_id, prediction.engine_name)
        device_id = self.get_device_id_for_sensor(prediction.sensor_id)

        target_ts = datetime.now(timezone.utc) + timedelta(
            minutes=prediction.horizon_steps * 10
        )

        row = self._conn.execute(
            text(
                """
                INSERT INTO dbo.predictions (
                  model_id, sensor_id, device_id,
                  predicted_value, confidence,
                  predicted_at, target_timestamp
                )
                OUTPUT INSERTED.id
                VALUES (
                  :model_id, :sensor_id, :device_id,
                  :predicted_value, :confidence,
                  GETDATE(), :target_timestamp
                )
                """
            ),
            {
                "model_id": model_id,
                "sensor_id": prediction.sensor_id,
                "device_id": device_id,
                "predicted_value": prediction.predicted_value,
                "confidence": prediction.confidence_score,
                "target_timestamp": target_ts.replace(tzinfo=None),
            },
        ).fetchone()

        if not row:
            raise RuntimeError("Failed to insert prediction")

        prediction_id = int(row[0])

        logger.debug(
            "storage_prediction_saved",
            extra={
                "prediction_id": prediction_id,
                "sensor_id": prediction.sensor_id,
                "engine": prediction.engine_name,
            },
        )

        return prediction_id

    def get_latest_prediction(self, sensor_id: int) -> Optional[Prediction]:
        """Obtiene la última predicción de un sensor."""
        row = self._conn.execute(
            text(
                """
                SELECT TOP 1
                  predicted_value, confidence
                FROM dbo.predictions
                WHERE sensor_id = :sensor_id
                ORDER BY predicted_at DESC
                """
            ),
            {"sensor_id": sensor_id},
        ).fetchone()

        if not row:
            return None

        return Prediction(
            sensor_id=sensor_id,
            predicted_value=_safe_float(row[0]),
            confidence_score=_safe_float(row[1]),
            trend="stable",
            engine_name="unknown",
        )

    # --- Anomalías ---

    def save_anomaly_event(
        self,
        anomaly: AnomalyResult,
        prediction_id: Optional[int] = None,
    ) -> int:
        """Persiste un evento de anomalía en dbo.ml_events."""
        device_id = self.get_device_id_for_sensor(anomaly.sensor_id)

        event_type = "warning" if anomaly.is_anomaly else "info"
        if anomaly.severity.value == "critical":
            event_type = "critical"

        row = self._conn.execute(
            text(
                """
                INSERT INTO dbo.ml_events (
                  device_id, sensor_id, prediction_id,
                  event_type, event_code, title, message,
                  status, created_at
                )
                OUTPUT INSERTED.id
                VALUES (
                  :device_id, :sensor_id, :prediction_id,
                  :event_type, 'ANOMALY_DETECTED', :title, :message,
                  'active', GETDATE()
                )
                """
            ),
            {
                "device_id": device_id,
                "sensor_id": anomaly.sensor_id,
                "prediction_id": prediction_id,
                "event_type": event_type,
                "title": f"Anomalía detectada: sensor {anomaly.sensor_id}",
                "message": anomaly.explanation,
            },
        ).fetchone()

        if not row:
            raise RuntimeError("Failed to insert anomaly event")

        return int(row[0])

    # --- Metadata ---

    def get_sensor_metadata(self, sensor_id: int) -> Dict[str, object]:
        """Obtiene metadata de un sensor."""
        row = self._conn.execute(
            text(
                """
                SELECT id, device_id, sensor_type, name, unit, location
                FROM dbo.sensors
                WHERE id = :sensor_id
                """
            ),
            {"sensor_id": sensor_id},
        ).fetchone()

        if not row:
            return {"sensor_id": sensor_id}

        return {
            "sensor_id": int(row[0]),
            "device_id": int(row[1]) if row[1] else 0,
            "sensor_type": str(row[2] or ""),
            "name": str(row[3] or ""),
            "unit": str(row[4] or ""),
            "location": str(row[5] or ""),
        }

    def get_device_id_for_sensor(self, sensor_id: int) -> int:
        """Obtiene el device_id de un sensor."""
        row = self._conn.execute(
            text("SELECT device_id FROM dbo.sensors WHERE id = :sensor_id"),
            {"sensor_id": sensor_id},
        ).fetchone()

        if not row:
            raise ValueError(f"Sensor {sensor_id} not found")

        return int(row[0])

    # --- Helpers privados ---

    def _get_or_create_model_id(self, sensor_id: int, engine_name: str) -> int:
        """Obtiene o crea un modelo activo para el sensor."""
        row = self._conn.execute(
            text(
                """
                SELECT TOP 1 id
                FROM dbo.ml_models
                WHERE sensor_id = :sensor_id AND is_active = 1
                ORDER BY trained_at DESC
                """
            ),
            {"sensor_id": sensor_id},
        ).fetchone()

        if row:
            return int(row[0])

        from iot_machine_learning.infrastructure.ml.engines.baseline_engine import BASELINE_MOVING_AVERAGE

        created = self._conn.execute(
            text(
                """
                INSERT INTO dbo.ml_models (
                  sensor_id, model_name, model_type, version,
                  is_active, trained_at
                )
                OUTPUT INSERTED.id
                VALUES (
                  :sensor_id, :model_name, :model_type, :version,
                  1, GETDATE()
                )
                """
            ),
            {
                "sensor_id": sensor_id,
                "model_name": BASELINE_MOVING_AVERAGE.name,
                "model_type": BASELINE_MOVING_AVERAGE.model_type,
                "version": BASELINE_MOVING_AVERAGE.version,
            },
        ).fetchone()

        if not created:
            raise RuntimeError("Failed to create ml_models row")

        return int(created[0])
