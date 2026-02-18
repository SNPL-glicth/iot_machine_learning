"""ml_stream — Facade del consumer de stream ML.

Expone start_consumer() como punto de entrada del procesador online.
"""

from __future__ import annotations


def start_consumer() -> None:
    """Inicia el consumer de stream ML (ReadingBroker → SlidingWindowBuffer).

    Bloquea hasta que el consumer se detenga o reciba señal de parada.
    """
    try:
        from iot_machine_learning.ml_service.consumers.stream_consumer import (
            ReadingsStreamConsumer,
        )
        consumer = ReadingsStreamConsumer()
        consumer.start()
    except Exception as exc:
        raise RuntimeError(f"Error iniciando stream consumer: {exc}") from exc


__all__ = ["start_consumer"]
