"""Buffer de eventos ML con flush periódico.

Extraído de MLEventPersister para modularidad (≤180 líneas).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class EventBuffer:
    """Buffer thread-safe para acumular eventos antes de flushear en batch.

    FIX P0-3: Reduce transacciones SQL al agrupar inserts.
    """

    def __init__(self, batch_size: int, flush_interval: float) -> None:
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._pending: List[dict] = []
        self._lock = threading.Lock()
        self._shutdown = False
        self._thread: Optional[threading.Thread] = None

        if batch_size > 1:
            self._thread = threading.Thread(
                target=self._flush_loop, daemon=True, name="EventBufferFlush"
            )
            self._thread.start()
            logger.info(
                "event_buffer_batch_enabled",
                extra={"batch_size": batch_size, "flush_interval": flush_interval},
            )
        else:
            logger.info("event_buffer_batch_disabled", extra={"batch_size": batch_size})

    def append(self, event_data: dict) -> bool:
        """Agrega un evento al buffer. Retorna True si se disparó flush."""
        with self._lock:
            self._pending.append(event_data)
            should_flush = len(self._pending) >= self._batch_size
            if should_flush:
                self._flush_locked()
            return should_flush

    def flush(self) -> List[dict]:
        """Fuerza flush y retorna los eventos extraídos."""
        with self._lock:
            return self._flush_locked()

    def _flush_locked(self) -> List[dict]:
        """Extrae y limpia el buffer. Debe llamarse con _lock adquirido."""
        if not self._pending:
            return []
        snapshot = self._pending[:]
        self._pending.clear()
        return snapshot

    def close(self) -> List[dict]:
        """Shutdown graceful: detiene el thread y retorna eventos pendientes."""
        self._shutdown = True
        logger.info("event_buffer_closing")
        return self.flush()

    def _flush_loop(self) -> None:
        """Daemon thread que flushea periódicamente."""
        logger.info("event_buffer_flush_thread_started")
        while not self._shutdown:
            time.sleep(self._flush_interval)
            try:
                flushed = self.flush()
                if flushed:
                    logger.info(
                        "event_buffer_periodic_flush",
                        extra={"flushed_events": len(flushed)},
                    )
            except Exception:
                logger.exception("event_buffer_flush_error")
