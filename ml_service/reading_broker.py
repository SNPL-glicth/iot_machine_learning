from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass(frozen=True)
class Reading:
    """Lectura cruda de sensor que viaja por el broker.

    Esta es la unidad mínima de datos que el ML online consume.
    """

    sensor_id: int
    sensor_type: str
    value: float
    timestamp: float  # epoch o device_ts normalizado


class ReadingBroker(Protocol):
    """Interfaz abstracta de stream/broker de lecturas.

    ML solo debe depender de esta interfaz, no de la implementación concreta.
    """

    def publish(self, reading: Reading) -> None:
        """Publicar una lectura en el stream.

        Implementaciones concretas decidirán si bloquean, descartan o aplican
        alguna política de backpressure. El core de ML no debe depender de eso.
        """

        ...

    def subscribe(self, handler: Callable[[Reading], None]) -> None:
        """Consumir lecturas de forma continua.

        Llama a ``handler(reading)`` por cada lectura recibida.
        Normalmente este método es bloqueante hasta que el proceso termina.
        """

        ...
