"""Raw SQL query helpers for ML event persistence.

Pure DB operations: no business logic, no threshold evaluation.
Extracted from EventWriter to separate SQL from business rules.
"""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)


def query_has_recent_delta_spike(
    conn: Connection,
    sensor_id: int,
    window_seconds: int = 30,
) -> bool:
    """Verifica si hay un DELTA_SPIKE activo/ack reciente.

    Args:
        conn: SQLAlchemy connection.
        sensor_id: Sensor identifier.
        window_seconds: Look-back window in seconds.

    Returns:
        True if a recent DELTA_SPIKE event exists.
    """
    row = conn.execute(
        text(
            """
            SELECT TOP 1 1
            FROM dbo.ml_events
            WHERE sensor_id = :sensor_id
              AND event_code = 'DELTA_SPIKE'
              AND status IN ('active', 'acknowledged')
              AND created_at >= DATEADD(second, -:sec, GETDATE())
            ORDER BY created_at DESC
            """
        ),
        {"sensor_id": sensor_id, "sec": window_seconds},
    ).fetchone()
    return row is not None


def query_upsert_event(
    conn: Connection,
    *,
    sensor_id: int,
    device_id: int,
    prediction_id: int,
    event_type: str,
    event_code: str,
    title: str,
    message: str,
    payload: str,
) -> tuple[bool, str]:
    """MERGE para ml_events: 1 evento activo por sensor + event_code.

    Args:
        conn: SQLAlchemy connection.
        sensor_id: Sensor identifier.
        device_id: Device identifier.
        prediction_id: Related prediction row ID.
        event_type: Event type string.
        event_code: Event code string.
        title: Human-readable title.
        message: Human-readable message.
        payload: JSON payload string.

    Returns:
        Tuple (is_new, action) where is_new=True means INSERT, False means UPDATE.
    """
    result = conn.execute(
        text(
            """
            DECLARE @existing_id INT, @action VARCHAR(10);

            SELECT TOP 1 @existing_id = id
            FROM dbo.ml_events
            WHERE sensor_id = :sensor_id
              AND event_code = :event_code
              AND status = 'active'
            ORDER BY created_at DESC;

            IF @existing_id IS NULL
            BEGIN
                INSERT INTO dbo.ml_events (
                    device_id, sensor_id, prediction_id,
                    event_type, event_code, title, message,
                    status, created_at, payload
                )
                VALUES (
                    :device_id, :sensor_id, :prediction_id,
                    :event_type, :event_code, :title, :message,
                    'active', GETDATE(), :payload
                );
                SET @action = 'INSERT';
            END
            ELSE
            BEGIN
                UPDATE dbo.ml_events
                SET device_id = :device_id,
                    prediction_id = :prediction_id,
                    event_type = :event_type,
                    title = :title,
                    message = :message,
                    created_at = GETDATE(),
                    payload = :payload
                WHERE id = @existing_id;
                SET @action = 'UPDATE';
            END

            SELECT @action AS action;
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
    ).fetchone()

    action = result[0] if result else "INSERT"
    return action == "INSERT", action


def query_resolve_event(
    conn: Connection,
    *,
    sensor_id: int,
    event_code: str,
) -> int:
    """UPDATE ml_events SET status='resolved' for active events.

    Args:
        conn: SQLAlchemy connection.
        sensor_id: Sensor identifier.
        event_code: Event code to resolve.

    Returns:
        Number of rows affected.
    """
    result = conn.execute(
        text(
            """
            UPDATE dbo.ml_events
            SET status = 'resolved',
                resolved_at = GETDATE()
            WHERE sensor_id = :sensor_id
              AND event_code = :event_code
              AND status = 'active'
            """
        ),
        {"sensor_id": sensor_id, "event_code": event_code},
    )
    return result.rowcount if hasattr(result, "rowcount") else 0


def query_active_threshold(conn: Connection, sensor_id: int):
    """Fetch the first active alert threshold for a sensor.

    Args:
        conn: SQLAlchemy connection.
        sensor_id: Sensor identifier.

    Returns:
        Row tuple (id, condition_type, threshold_value_min,
        threshold_value_max, severity, name) or None.
    """
    return conn.execute(
        text(
            """
            SELECT TOP 1
              id, condition_type, threshold_value_min, threshold_value_max, severity, name
            FROM dbo.alert_thresholds
            WHERE sensor_id = :sensor_id AND is_active = 1
            ORDER BY id ASC
            """
        ),
        {"sensor_id": sensor_id},
    ).fetchone()
