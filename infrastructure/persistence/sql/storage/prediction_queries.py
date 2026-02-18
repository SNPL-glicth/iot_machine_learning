"""Prediction queries module."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Connection

from .....domain.entities.prediction import Prediction
from .....domain.validators.input_guard import safe_series_id_to_int

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


class PredictionQueries:
    """Queries para predicciones."""
    
    def __init__(self, conn: Connection) -> None:
        self._conn = conn
    
    def save_prediction(
        self,
        prediction: Prediction,
        get_device_id_fn,
        *,
        horizon_minutes_per_step: int = 10,
    ) -> int:
        """Persiste una predicción en dbo.predictions."""
        _sensor_id = safe_series_id_to_int(prediction.series_id)
        model_id = self._get_or_create_model_id(_sensor_id, prediction.engine_name)
        device_id = get_device_id_fn(_sensor_id)

        target_ts = datetime.now(timezone.utc) + timedelta(
            minutes=prediction.horizon_steps * horizon_minutes_per_step
        )

        meta = prediction.metadata or {}
        _is_anomaly = 1 if meta.get("is_anomaly") else 0
        _anomaly_score = _safe_float(meta.get("anomaly_score"), default=0.0) if meta.get("anomaly_score") is not None else None
        _severity = str(meta.get("severity", "info"))
        if _severity not in ("info", "warning", "critical"):
            _severity = "info"
        _risk_level = str(meta.get("risk_level", "NONE"))
        if _risk_level not in ("NONE", "LOW", "MEDIUM", "HIGH"):
            _risk_level = "NONE"
        _explanation = str(meta.get("explanation", "")) or None
        _horizon_minutes = prediction.horizon_steps * horizon_minutes_per_step

        row = self._conn.execute(
            text(
                """
                INSERT INTO dbo.predictions (
                  model_id, sensor_id, device_id,
                  predicted_value, confidence,
                  predicted_at, target_timestamp,
                  horizon_minutes, window_points,
                  engine_name, trend,
                  is_anomaly, anomaly_score,
                  risk_level, severity,
                  explanation, status
                )
                OUTPUT INSERTED.id
                VALUES (
                  :model_id, :sensor_id, :device_id,
                  :predicted_value, :confidence,
                  GETDATE(), :target_timestamp,
                  :horizon_minutes, :window_points,
                  :engine_name, :trend,
                  :is_anomaly, :anomaly_score,
                  :risk_level, :severity,
                  :explanation, 'active'
                )
                """
            ),
            {
                "model_id": model_id,
                "sensor_id": _sensor_id,
                "device_id": device_id,
                "predicted_value": prediction.predicted_value,
                "confidence": prediction.confidence_score,
                "target_timestamp": target_ts.replace(tzinfo=None),
                "horizon_minutes": _horizon_minutes,
                "window_points": 0,
                "engine_name": prediction.engine_name,
                "trend": prediction.trend,
                "is_anomaly": _is_anomaly,
                "anomaly_score": _anomaly_score,
                "risk_level": _risk_level,
                "severity": _severity,
                "explanation": _explanation,
            },
        ).fetchone()

        if not row:
            raise RuntimeError("Failed to insert prediction")

        prediction_id = int(row[0])

        logger.info(
            "storage_prediction_saved",
            extra={
                "prediction_id": prediction_id,
                "sensor_id": _sensor_id,
                "engine": prediction.engine_name,
                "trend": prediction.trend,
                "confidence": prediction.confidence_score,
            },
        )

        return prediction_id

    def get_latest_prediction(self, sensor_id: int) -> Optional[Prediction]:
        """Obtiene la última predicción de un sensor."""
        row = self._conn.execute(
            text(
                """
                SELECT TOP 1
                  predicted_value, confidence,
                  engine_name, trend
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
            series_id=str(sensor_id),
            predicted_value=_safe_float(row[0]),
            confidence_score=_safe_float(row[1]),
            trend=str(row[2] or "stable"),
            engine_name=str(row[3] or "unknown"),
        )
    
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
                "model_name": engine_name,
                "model_type": "prediction",
                "version": "1.0",
            },
        ).fetchone()

        if not created:
            raise RuntimeError("Failed to create ml_models row")

        logger.info(
            "storage_model_created",
            extra={
                "sensor_id": sensor_id,
                "engine_name": engine_name,
                "model_id": int(created[0]),
            },
        )

        return int(created[0])
