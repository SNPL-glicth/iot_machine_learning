from __future__ import annotations

import queue
from typing import Callable

from .reading_broker import Reading, ReadingBroker


class InMemoryReadingBroker(ReadingBroker):
    """Implementación sencilla en memoria para el stream de lecturas.

    - Un único productor principal (ingest_api) que publica lecturas.
    - Un único consumidor (runner de ML online) que se subscribe.
    - Sin asyncio, sin múltiples consumidores, sin locks explícitos.
    """

    def __init__(self, maxsize: int = 100_000) -> None:
        self._queue: "queue.Queue[Reading]" = queue.Queue(maxsize=maxsize)
        self._stopped: bool = False

    def publish(self, reading: Reading) -> None:  # type: ignore[override]
        """Encola la lectura para ser consumida por ML.

        En caso de cola llena, se descarta silenciosamente en este MVP.
        """

        try:
            # No bloqueamos para evitar acoplar ML al ritmo de ingestión.
            self._queue.put(reading, block=False)
        except queue.Full:
            # Aquí podrías loggear una métrica simple si lo necesitas.
            pass

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
