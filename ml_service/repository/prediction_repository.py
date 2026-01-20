from __future__ import annotations

from datetime import datetime

from sqlalchemy import text
from sqlalchemy.engine import Connection


def get_or_create_model_id(conn: Connection, sensor_id: int) -> int:
    row = conn.execute(
        text(
            """
            SELECT TOP 1 id
            FROM dbo.ml_models
            WHERE sensor_id = :sensor_id AND is_active = 1
              AND model_type = 'sklearn'
            ORDER BY trained_at DESC
            """
        ),
        {"sensor_id": sensor_id},
    ).fetchone()

    if row:
        return int(row[0])

    created = conn.execute(
        text(
            """
            INSERT INTO dbo.ml_models (sensor_id, model_name, model_type, version, is_active, trained_at)
            OUTPUT INSERTED.id
            VALUES (:sensor_id, :model_name, :model_type, :version, 1, GETDATE())
            """
        ),
        {
            "sensor_id": sensor_id,
            "model_name": "sklearn_regression_iforest",
            "model_type": "sklearn",
            "version": "1.0.0",
        },
    ).fetchone()

    if not created:
        raise RuntimeError("failed to create ml_models row")

    return int(created[0])


def insert_prediction(
    conn: Connection,
    *,
    model_id: int,
    sensor_id: int,
    predicted_value: float,
    confidence: float,
    target_ts_utc: datetime,
) -> int:
    row = conn.execute(
        text(
            """
            INSERT INTO dbo.predictions (
              model_id, sensor_id, predicted_value, confidence, predicted_at, target_timestamp
            )
            OUTPUT INSERTED.id
            VALUES (
              :model_id, :sensor_id, :predicted_value, :confidence, GETDATE(), :target_timestamp
            )
            """
        ),
        {
            "model_id": model_id,
            "sensor_id": sensor_id,
            "predicted_value": predicted_value,
            "confidence": confidence,
            "target_timestamp": target_ts_utc.replace(tzinfo=None),
        },
    ).fetchone()

    if not row:
        raise RuntimeError("failed to insert prediction")

    return int(row[0])


def insert_anomaly_event(
    conn: Connection,
    *,
    device_id: int,
    sensor_id: int,
    prediction_id: int,
    anomaly_score: float,
    trend: str,
    explanation: str,
) -> None:
    event_code = "ANOMALY_DETECTED"
    event_type = "warning"

    title = "Posible anomalía detectada por ML"
    message = f"anomaly_score={anomaly_score:.4f} trend={trend}"
    payload = (
        "{"  # JSON simple para trazabilidad
        f"\"anomaly_score\": {anomaly_score:.4f}, "
        f"\"trend\": \"{trend}\", "
        f"\"explanation\": \"{explanation}\""
        "}"
    )

    conn.execute(
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
            "event_type": event_type,
            "event_code": event_code,
            "title": title,
            "message": message,
            "payload": payload,
        },
    )
