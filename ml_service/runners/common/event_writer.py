"""Escritor de eventos ML.

Responsabilidad única: lógica de negocio para eventos ML.
Las queries SQL están en event_queries.py.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from sqlalchemy.engine import Connection

from .event_queries import (
    query_active_threshold,
    query_has_recent_delta_spike,
    query_resolve_event,
    query_upsert_event,
)

if TYPE_CHECKING:
    from iot_machine_learning.ml_service.explain.explanation_builder import PredictionExplanation

logger = logging.getLogger(__name__)


class EventWriter:
    """Gestiona eventos ML en la base de datos.

    Responsabilidades:
    - Lógica de negocio: cuándo emitir/resolver eventos
    - Delega queries SQL a event_queries.py
    """

    def can_sensor_emit_events(self, conn: Connection, sensor_id: int) -> bool:
        """Verifica si el sensor puede emitir eventos ML.

        REGLA: Sensores en INITIALIZING o STALE NO pueden generar eventos.
        """
        try:
            from iot_ingest_services.ingest_api.sensor_state import SensorStateManager
            state_manager = SensorStateManager(conn)
            can_generate, reason = state_manager.can_generate_events(sensor_id)
            if not can_generate:
                logger.debug(
                    "[EVENT_WRITER] sensor_id=%s cannot emit events: %s",
                    sensor_id, reason
                )
            return can_generate
        except Exception as e:
            logger.warning(
                "[EVENT_WRITER] sensor_id=%s state check error=%s, allowing events",
                sensor_id, str(e)
            )
            return True

    def has_recent_delta_spike(
        self,
        conn: Connection,
        sensor_id: int,
        window_seconds: int = 30,
    ) -> bool:
        """Verifica si hay un DELTA_SPIKE activo/ack reciente."""
        return query_has_recent_delta_spike(conn, sensor_id, window_seconds)

    def upsert_event(
        self,
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
        """MERGE para ml_events: 1 evento activo por sensor + event_code."""
        return query_upsert_event(
            conn,
            sensor_id=sensor_id,
            device_id=device_id,
            prediction_id=prediction_id,
            event_type=event_type,
            event_code=event_code,
            title=title,
            message=message,
            payload=payload,
        )

    def resolve_event_if_exists(
        self,
        conn: Connection,
        *,
        sensor_id: int,
        event_code: str,
    ) -> bool:
        """Resuelve evento activo si existe (transición a NORMAL)."""
        affected = query_resolve_event(conn, sensor_id=sensor_id, event_code=event_code)
        if affected > 0:
            logger.info(
                "[EVENT_WRITER] sensor_id=%s event_code=%s resolved",
                sensor_id, event_code
            )
            try:
                from iot_ingest_services.ingest_api.sensor_state import SensorStateManager
                state_manager = SensorStateManager(conn)
                state_manager.sync_state_with_events(sensor_id)
            except Exception as e:
                logger.warning("[EVENT_WRITER] state sync error: %s", str(e))
        return affected > 0

    def insert_threshold_event_if_needed(
        self,
        conn: Connection,
        *,
        sensor_id: int,
        device_id: int,
        prediction_id: int,
        predicted_value: float,
    ) -> None:
        """Inserta evento de violación de umbral si corresponde."""
        if not self.can_sensor_emit_events(conn, sensor_id):
            return

        thr = query_active_threshold(conn, sensor_id)
        if not thr:
            return

        threshold_id, cond, vmin, vmax, severity, thr_name = thr
        vmin_f = float(vmin) if vmin is not None else None
        vmax_f = float(vmax) if vmax is not None else None
        violated = (
            (cond == "greater_than" and vmin_f is not None and predicted_value > vmin_f)
            or (cond == "less_than" and vmin_f is not None and predicted_value < vmin_f)
            or (cond == "out_of_range" and vmin_f is not None and vmax_f is not None
                and (predicted_value < vmin_f or predicted_value > vmax_f))
            or (cond == "equal_to" and vmin_f is not None and predicted_value == vmin_f)
        )

        event_code = "PRED_THRESHOLD_BREACH"
        if not violated:
            self.resolve_event_if_exists(conn, sensor_id=sensor_id, event_code=event_code)
            return

        sev = str(severity)
        event_type = "critical" if sev == "critical" else "warning" if sev == "warning" else "notice"
        payload = json.dumps({
            "threshold_id": int(threshold_id),
            "condition_type": cond,
            "threshold_value_min": float(vmin) if vmin is not None else None,
            "threshold_value_max": float(vmax) if vmax is not None else None,
            "predicted_value": predicted_value,
        }, ensure_ascii=False)

        _, action = self.upsert_event(
            conn,
            sensor_id=sensor_id, device_id=device_id, prediction_id=prediction_id,
            event_type=event_type, event_code=event_code,
            title=f"Predicción viola umbral: {thr_name}",
            message=f"predicted_value={predicted_value:.4f} threshold_id={int(threshold_id)}",
            payload=payload,
        )
        logger.info(
            "[EVENT_WRITER] threshold event sensor_id=%s action=%s predicted=%.4f",
            sensor_id, action, predicted_value
        )

    def insert_anomaly_event(
        self,
        conn: Connection,
        *,
        sensor_id: int,
        device_id: int,
        prediction_id: int,
        explanation: "PredictionExplanation",
        severity_classifier: "SeverityClassifier",
    ) -> None:
        """Inserta evento de anomalía si corresponde."""
        event_code = "ANOMALY_DETECTED"
        if not explanation.anomaly:
            self.resolve_event_if_exists(conn, sensor_id=sensor_id, event_code=event_code)
            return

        if severity_classifier.is_value_within_user_thresholds(
            conn, sensor_id, explanation.predicted_value
        ):
            logger.debug(
                "[EVENT_WRITER] anomaly suppressed sensor_id=%s (within user thresholds)",
                sensor_id
            )
            self.resolve_event_if_exists(conn, sensor_id=sensor_id, event_code=event_code)
            return

        if not self.can_sensor_emit_events(conn, sensor_id):
            return

        payload = json.dumps({
            "severity": explanation.severity,
            "action_required": explanation.action_required,
            "anomaly_score": explanation.anomaly_score,
            "trend": explanation.trend,
            "predicted_value": explanation.predicted_value,
            "confidence": explanation.confidence,
            "recommended_action": explanation.recommended_action,
            "explanation": explanation.explanation,
        }, ensure_ascii=False)

        _, action = self.upsert_event(
            conn,
            sensor_id=sensor_id, device_id=device_id, prediction_id=prediction_id,
            event_type="warning", event_code=event_code,
            title="Posible anomalía detectada por ML",
            message=(
                f"severidad={explanation.severity} "
                f"action_required={explanation.action_required} "
                f"trend={explanation.trend}"
            ),
            payload=payload,
        )
        logger.info(
            "[EVENT_WRITER] anomaly event sensor_id=%s action=%s score=%.4f",
            sensor_id, action, explanation.anomaly_score
        )


# Import para type hints (al final para evitar circular imports)
try:
    from .severity_classifier import SeverityClassifier
except ImportError:
    from severity_classifier import SeverityClassifier
