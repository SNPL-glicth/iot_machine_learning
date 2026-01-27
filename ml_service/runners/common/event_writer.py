"""Escritor de eventos ML.

Responsabilidad única: Gestionar eventos ML en la base de datos.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.engine import Connection

if TYPE_CHECKING:
    from iot_machine_learning.ml_service.explain.explanation_builder import PredictionExplanation

logger = logging.getLogger(__name__)


class EventWriter:
    """Gestiona eventos ML en la base de datos.
    
    Responsabilidades:
    - Insertar/actualizar eventos (MERGE)
    - Resolver eventos cuando vuelven a normal
    - Verificar estado operacional del sensor
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
        window_seconds: int = 30
    ) -> bool:
        """Verifica si hay un DELTA_SPIKE activo/ack reciente."""
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
        """MERGE para ml_events: 1 evento activo por sensor + event_code.
        
        Returns:
            (is_new, action): True si se insertó, False si se actualizó
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
        is_new = action == "INSERT"
        return is_new, action
    
    def resolve_event_if_exists(
        self,
        conn: Connection,
        *,
        sensor_id: int,
        event_code: str,
    ) -> bool:
        """Resuelve evento activo si existe (transición a NORMAL).
        
        Returns:
            True si se resolvió un evento
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
        
        affected = result.rowcount if hasattr(result, 'rowcount') else 0
        if affected > 0:
            logger.info(
                "[EVENT_WRITER] sensor_id=%s event_code=%s resolved",
                sensor_id, event_code
            )
            # Sincronizar estado del sensor
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
        
        thr = conn.execute(
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

        if not thr:
            return

        threshold_id, cond, vmin, vmax, severity, thr_name = thr
        vmin_f = float(vmin) if vmin is not None else None
        vmax_f = float(vmax) if vmax is not None else None
        violated = False

        if cond == "greater_than" and vmin_f is not None and predicted_value > vmin_f:
            violated = True
        elif cond == "less_than" and vmin_f is not None and predicted_value < vmin_f:
            violated = True
        elif cond == "out_of_range" and vmin_f is not None and vmax_f is not None:
            violated = predicted_value < vmin_f or predicted_value > vmax_f
        elif cond == "equal_to" and vmin_f is not None and predicted_value == vmin_f:
            violated = True

        event_code = "PRED_THRESHOLD_BREACH"
        
        if not violated:
            self.resolve_event_if_exists(conn, sensor_id=sensor_id, event_code=event_code)
            return

        sev = str(severity)
        event_type = "critical" if sev == "critical" else "warning" if sev == "warning" else "notice"

        title = f"Predicción viola umbral: {thr_name}"
        message = f"predicted_value={predicted_value:.4f} threshold_id={int(threshold_id)}"

        payload = json.dumps({
            "threshold_id": int(threshold_id),
            "condition_type": cond,
            "threshold_value_min": float(vmin) if vmin is not None else None,
            "threshold_value_max": float(vmax) if vmax is not None else None,
            "predicted_value": predicted_value,
        }, ensure_ascii=False)

        is_new, action = self.upsert_event(
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
        
        # Verificar si el valor está dentro del rango del usuario
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

        event_type = "warning"
        title = "Posible anomalía detectada por ML"
        message = (
            f"severidad={explanation.severity} "
            f"action_required={explanation.action_required} "
            f"trend={explanation.trend}"
        )

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

        is_new, action = self.upsert_event(
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
        
        logger.info(
            "[EVENT_WRITER] anomaly event sensor_id=%s action=%s score=%.4f",
            sensor_id, action, explanation.anomaly_score
        )


# Import para type hints (al final para evitar circular imports)
try:
    from .severity_classifier import SeverityClassifier
except ImportError:
    from severity_classifier import SeverityClassifier
