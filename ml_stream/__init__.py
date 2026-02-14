"""ML Stream Consumer facade (E-13).

Re-exports from ml_service.consumers for independent deployment.
Existing imports via ml_service.consumers continue to work unchanged.

Usage:
    from ml_stream import start_consumer
    start_consumer()
"""

from __future__ import annotations


def start_consumer(redis_url: str | None = None):
    """Start the readings stream consumer."""
    from iot_machine_learning.ml_service.consumers.stream_consumer import (
        ReadingsStreamConsumer,
    )
    consumer = ReadingsStreamConsumer(redis_url=redis_url)
    consumer.start()
