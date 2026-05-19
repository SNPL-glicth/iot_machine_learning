"""Event persister service for ML online processing.

Orquestador delegado a event_inserter (SQL puro) y event_buffer (batching).
Mantiene retrocompatibilidad total de la interfaz pública.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from sqlalchemy import text
from sqlalchemy.engine import Connection

from .event_buffer import EventBuffer
from .event_inserter import insert_single, insert_single_transaction
from ..models.online_analysis import OnlineAnalysis

logger = logging.getLogger(__name__)


class MLEventPersister:
    """Persiste eventos ML y notificaciones en la base de datos.

    FIX P0-3: Batching de eventos en buffer con flush periódico.
    Responsabilidades: delega inserts a event_inserter y buffering a event_buffer.
    """

    def __init__(self) -> None:
        self._device_cache: Dict[int, int] = {}

        from os import getenv
        batch_size = int(getenv("ML_EVENT_BATCH_SIZE", "50"))
        flush_interval = float(getenv("ML_EVENT_FLUSH_INTERVAL_SECONDS", "5.0"))
        self._buffer = EventBuffer(batch_size, flush_interval) if batch_size > 1 else None
        self._batch_size = batch_size

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
        """Inserta un ml_event. batch_size=1 → inmediato; >1 → buffer."""
        event_data = {
            "sensor_id": sensor_id,
            "sensor_type": sensor_type,
            "severity_label": severity_label,
            "event_type": event_type,
            "event_code": event_code,
            "title": title,
            "explanation": explanation,
            "recommended_action": recommended_action,
            "analysis": analysis,
            "ts_utc": ts_utc,
            "prediction_id": prediction_id,
            "extra_payload": extra_payload,
        }

        if self._buffer is None:
            insert_single_transaction(
                **event_data, device_cache=self._device_cache
            )
            return

        self._buffer.append(event_data)

    def flush(self) -> int:
        """Fuerza flush inmediato. Retorna eventos flusheados."""
        if self._buffer is None:
            return 0
        buffer = self._buffer.flush()
        if not buffer:
            return 0
        return self._flush_batch(buffer)

    def close(self) -> int:
        """Shutdown graceful: retorna eventos pendientes flusheados."""
        if self._buffer is None:
            return 0
        buffer = self._buffer.close()
        return self._flush_batch(buffer)

    def _flush_batch(self, buffer: list) -> int:
        """Flush real: batch → fallback individual."""
        count = len(buffer)
        logger.info("ml_event_persister_flush_start", extra={"buffered_events": count})

        # Batch en una sola transacción
        from iot_ingest_services.common.db import get_engine
        try:
            engine = get_engine()
            with engine.begin() as conn:
                for event in buffer:
                    insert_single(
                        conn, **event, device_cache=self._device_cache
                    )
            logger.info("ml_event_persister_flush_success", extra={"flushed_events": count})
            return count
        except Exception:
            logger.exception("ml_event_persister_batch_failed", extra={"buffered_events": count})

        # Fallback individual
        ok = 0
        for event in buffer:
            try:
                insert_single_transaction(**event, device_cache=self._device_cache)
                ok += 1
            except Exception:
                logger.exception(
                    "ml_event_persister_individual_fallback_failed",
                    extra={
                        "sensor_id": event.get("sensor_id"),
                        "event_code": event.get("event_code"),
                    },
                )
        logger.warning(
            "ml_event_persister_fallback_complete",
            extra={"buffered_events": count, "fallback_ok": ok, "fallback_failed": count - ok},
        )
        return ok

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
