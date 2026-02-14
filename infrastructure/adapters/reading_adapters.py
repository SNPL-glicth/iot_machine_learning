"""Adapters to convert service-specific reading types to canonical SensorReading (E-6).

The ML domain's SensorReading is the canonical type. Each service has its own
reading representation; these adapters convert FROM those representations TO
SensorReading without modifying the source services.

Supported conversions:
- stream Reading (ml_service/consumers/sliding_window.py)
- broker Reading (iot_broker/domain/entities/reading.py)
- reading_broker Reading (ml_service/reading_broker.py)
- MQTTReadingPayload (iot_ingest_services/ingest_api/mqtt/validators.py)
- UnifiedReading (iot_ingest_services/ingest_api/pipelines/contracts/)
- raw dict (Redis stream fields, JSON payloads)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from iot_machine_learning.domain.entities.iot.sensor_reading import SensorReading

logger = logging.getLogger(__name__)


def from_stream_reading(reading: Any) -> SensorReading:
    """Convert ml_service.consumers.sliding_window.Reading → SensorReading.

    The stream Reading has: sensor_id, value, timestamp, timestamp_iso.
    """
    return SensorReading(
        sensor_id=reading.sensor_id,
        value=reading.value,
        timestamp=reading.timestamp,
    )


def from_broker_reading(reading: Any) -> SensorReading:
    """Convert iot_broker.domain.entities.reading.Reading → SensorReading.

    The broker Reading has: sensor_id, value, timestamp, device_uuid,
    sensor_uuid, device_timestamp, msg_id.
    """
    return SensorReading(
        sensor_id=reading.sensor_id,
        value=reading.value,
        timestamp=reading.timestamp,
    )


def from_reading_broker(reading: Any) -> SensorReading:
    """Convert ml_service.reading_broker.Reading → SensorReading.

    The reading_broker Reading has: sensor_id, sensor_type, value, timestamp.
    """
    return SensorReading(
        sensor_id=reading.sensor_id,
        value=reading.value,
        timestamp=reading.timestamp,
        sensor_type=getattr(reading, "sensor_type", ""),
    )


def from_mqtt_payload(payload: Any) -> Optional[SensorReading]:
    """Convert MQTTReadingPayload → SensorReading.

    Returns None if sensor_id is not numeric.
    """
    sid = payload.sensor_id_int
    if sid is None:
        logger.warning("Cannot convert MQTTReadingPayload: non-numeric sensor_id=%s", payload.sensor_id)
        return None
    return SensorReading(
        sensor_id=sid,
        value=payload.value,
        timestamp=payload.timestamp_float,
        sensor_type=payload.sensor_type or "",
    )


def from_dict(data: Dict[str, Any]) -> SensorReading:
    """Convert a raw dict (e.g. Redis stream fields) → SensorReading.

    Expected keys: sensor_id, value, timestamp.
    Optional: sensor_type, device_id.
    """
    return SensorReading(
        sensor_id=int(data["sensor_id"]),
        value=float(data["value"]),
        timestamp=float(data["timestamp"]),
        sensor_type=str(data.get("sensor_type", "")),
        device_id=int(data["device_id"]) if data.get("device_id") else None,
    )
