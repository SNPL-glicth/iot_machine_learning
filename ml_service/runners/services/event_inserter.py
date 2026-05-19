"""Inserción pura de eventos ML en la base de datos.

Funciones stateless que operan sobre una conexión SQL existente.
Extraídas de MLEventPersister para modularidad (≤180 líneas).
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


def _get_device_id(conn: Connection, sensor_id: int, device_cache: Dict[int, int]) -> int:
    if sensor_id in device_cache:
        return device_cache[sensor_id]
    device_id = get_device_id_for_sensor(conn, sensor_id)
    device_cache[sensor_id] = device_id
    return device_id


def insert_single(
    conn: Connection,
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
    device_cache: Dict[int, int],
) -> None:
    """Inserta un evento ML y su notificación usando una conexión existente."""
    recent = conn.execute(
        text(
            "SELECT TOP 1 1 FROM dbo.ml_events "
            "WHERE sensor_id = :sensor_id AND event_code = :event_code "
            "AND created_at >= DATEADD(MINUTE, -5, GETDATE())"
        ),
        {"sensor_id": sensor_id, "event_code": event_code},
    ).fetchone()
    if recent:
        logger.debug("[ML_COOLDOWN] sensor_id=%s event_code=%s skipping", sensor_id, event_code)
        return

    active = conn.execute(
        text(
            "SELECT TOP 1 1 FROM dbo.ml_events "
            "WHERE sensor_id = :sensor_id AND event_code = :event_code AND status = 'active'"
        ),
        {"sensor_id": sensor_id, "event_code": event_code},
    ).fetchone()
    if active:
        logger.debug("[ML_DEDUPE] sensor_id=%s event_code=%s skipping", sensor_id, event_code)
        return

    device_id = _get_device_id(conn, sensor_id, device_cache)
    base: dict = {
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
        base.update(extra_payload)
    payload_json = json.dumps(base, ensure_ascii=False)

    row = conn.execute(
        text(
            "INSERT INTO dbo.ml_events (device_id, sensor_id, prediction_id, event_type, "
            "event_code, title, message, status, created_at, payload) "
            "OUTPUT INSERTED.id VALUES (:device_id, :sensor_id, :prediction_id, "
            ":event_type, :event_code, :title, :message, 'active', "
            "DATEADD(second, :ts_utc, '1970-01-01'), :payload)"
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

    try:
        conn.execute(
            text(
                "INSERT INTO dbo.alert_notifications (source, source_event_id, severity, "
                "title, message, is_read, created_at) VALUES (:source, :source_event_id, "
                ":severity, :title, :message, 0, GETDATE())"
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
        logger.exception("alert_notifications insert failed (ML online)")


def insert_single_transaction(
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
    device_cache: Dict[int, int],
) -> None:
    """Inserta un evento en su propia transacción (comportamiento original)."""
    engine = get_engine()
    with engine.begin() as conn:
        insert_single(
            conn,
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            severity_label=severity_label,
            event_type=event_type,
            event_code=event_code,
            title=title,
            explanation=explanation,
            recommended_action=recommended_action,
            analysis=analysis,
            ts_utc=ts_utc,
            prediction_id=prediction_id,
            extra_payload=extra_payload,
            device_cache=device_cache,
        )
