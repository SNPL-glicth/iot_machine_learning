"""Anomaly queries module."""

from __future__ import annotations

import json
import logging
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Connection

from iot_machine_learning.domain.entities.anomaly import AnomalyResult
from iot_machine_learning.domain.validators.input_guard import safe_series_id_to_int

logger = logging.getLogger(__name__)


class AnomalyQueries:
    """Queries para anomalías."""
    
    def __init__(self, conn: Connection) -> None:
        self._conn = conn
    
    def save_anomaly_event(
        self,
        anomaly: AnomalyResult,
        get_device_id_fn,
        prediction_id: Optional[int] = None,
    ) -> int:
        """Persiste un evento de anomalía en dbo.ml_events."""
        _sensor_id = safe_series_id_to_int(anomaly.series_id)
        device_id = get_device_id_fn(_sensor_id)

        event_type = "warning" if anomaly.is_anomaly else "info"
        if anomaly.severity.value == "critical":
            event_type = "critical"

        votes_json = json.dumps(
            {k: round(v, 6) for k, v in anomaly.method_votes.items()}
        ) if anomaly.method_votes else None

        row = self._conn.execute(
            text(
                """
                INSERT INTO dbo.ml_events (
                  device_id, sensor_id, prediction_id,
                  event_type, event_code, title, message,
                  anomaly_score, anomaly_confidence,
                  method_votes, audit_trace_id,
                  status, created_at
                )
                OUTPUT INSERTED.id
                VALUES (
                  :device_id, :sensor_id, :prediction_id,
                  :event_type, 'ANOMALY_DETECTED', :title, :message,
                  :anomaly_score, :anomaly_confidence,
                  :method_votes, :audit_trace_id,
                  'active', GETDATE()
                )
                """
            ),
            {
                "device_id": device_id,
                "sensor_id": _sensor_id,
                "prediction_id": prediction_id,
                "event_type": event_type,
                "title": f"Anomalía detectada: serie {anomaly.series_id}",
                "message": anomaly.explanation,
                "anomaly_score": round(anomaly.score, 6),
                "anomaly_confidence": round(anomaly.confidence, 6),
                "method_votes": votes_json,
                "audit_trace_id": anomaly.audit_trace_id,
            },
        ).fetchone()

        if not row:
            raise RuntimeError("Failed to insert anomaly event")

        event_id = int(row[0])

        logger.info(
            "storage_anomaly_event_saved",
            extra={
                "event_id": event_id,
                "sensor_id": _sensor_id,
                "event_type": event_type,
                "score": anomaly.score,
                "confidence": anomaly.confidence,
                "audit_trace_id": anomaly.audit_trace_id,
            },
        )

        return event_id
