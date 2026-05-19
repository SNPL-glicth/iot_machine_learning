from __future__ import annotations

import logging
import queue
import time
from typing import Callable

from .reading_broker import Reading, ReadingBroker

logger = logging.getLogger(__name__)


class InMemoryReadingBroker(ReadingBroker):
    """Implementación sencilla en memoria para el stream de lecturas.

    - Un único productor principal (ingest_api) que publica lecturas.
    - Un único consumidor (runner de ML online) que se subscribe.
    - Sin asyncio, sin múltiples consumidores, sin locks explícitos.
    """

    def __init__(self, maxsize: int = 100_000) -> None:
        self._queue: "queue.Queue[Reading]" = queue.Queue(maxsize=maxsize)
        self._stopped: bool = False
        self._drop_count: int = 0
        self._last_drop_summary: float = 0.0
        # BACKLOG: Drop silencioso reemplazado por log estructurado.
        # Backpressure real (bloqueo o rechazo con HTTP 429) requiere
        # rediseño del broker — ver BACKLOG arquitectural.

    @property
    def drop_count(self) -> int:
        """Total de mensajes descartados por cola llena."""
        return self._drop_count

    def publish(self, reading: Reading) -> None:  # type: ignore[override]
        """Encola la lectura para ser consumida por ML.

        En caso de cola llena, se descarta con WARNING estructurado.
        """

        try:
            # No bloqueamos para evitar acoplar ML al ritmo de ingestión.
            self._queue.put(reading, block=False)
        except queue.Full:
            self._drop_count += 1
            logger.warning(
                {"event": "broker_queue_full_drop",
                 "sensor_id": getattr(reading, "sensor_id", "unknown"),
                 "queue_size": self._queue.maxsize,
                 "total_drops": self._drop_count}
            )
            # Resumen periódico cada 60s
            now = time.monotonic()
            if now - self._last_drop_summary > 60.0:
                self._last_drop_summary = now
                logger.warning(
                    {"event": "broker_drop_summary",
                     "total_drops": self._drop_count,
                     "queue_maxsize": self._queue.maxsize}
                )

    def subscribe(self, handler: Callable[[Reading], None]) -> None:  # type: ignore[override]
        """Bucle bloqueante que entrega lecturas al handler.

        Diseñado para tener un solo consumidor (el proceso de ML online).
        """

        while not self._stopped:
            try:
                reading = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                handler(reading)
            finally:
                self._queue.task_done()

    def stop(self) -> None:
        """Señala que no se aceptarán más elementos.

        No implementamos una parada sofisticada; el consumidor saldrá
        cuando se procese la cola restante y `_stopped` sea True.
        """

        self._stopped = True
