"""Adapter IoT → Zenin: traduce sensor_id:int a series_id:UUID.

Este es el ÚNICO punto de traducción entre el modelo IoT legacy y Zenin.
Toda conversión sensor_id → series_id ocurre aquí.

Responsabilidades:
- Conversión determinística sensor_id:int → series_id:UUID
- Transformación SensorReading → DataPoint
- Transformación List[SensorReading] → TimeWindow
- Preservación de trazabilidad (sensor_id en metadata)
"""

from __future__ import annotations

from uuid import UUID, uuid5, NAMESPACE_OID
from typing import List

from iot_machine_learning.domain.entities.series import DataPoint, TimeWindow
from iot_machine_learning.domain.entities.iot.sensor_reading import (
    Reading,
    SensorReading,
)

# Namespace fijo para generación determinística de UUIDs desde sensor_id
# Usar NAMESPACE_OID garantiza que los UUIDs sean reproducibles
_SENSOR_NAMESPACE = uuid5(NAMESPACE_OID, "zenin.iot.sensor")


def sensor_id_to_series_id(sensor_id: int, tenant_id: UUID) -> UUID:
    """Convierte sensor_id:int a series_id:UUID de forma determinística.
    
    La conversión es idempotente: mismo sensor_id + tenant_id → mismo UUID.
    Permite trazabilidad sin tabla de mapeo en DB.
    
    Args:
        sensor_id: ID numérico del sensor (legacy IoT)
        tenant_id: UUID del tenant (multi-tenancy)
    
    Returns:
        UUID determinístico para la serie
    
    Examples:
        >>> tenant = UUID("12345678-1234-5678-1234-567812345678")
        >>> uuid1 = sensor_id_to_series_id(42, tenant)
        >>> uuid2 = sensor_id_to_series_id(42, tenant)
        >>> uuid1 == uuid2
        True
    """
    # Combinar tenant_id y sensor_id en un string único
    composite_key = f"{tenant_id}:{sensor_id}"
    
    # Generar UUID determinístico usando uuid5
    return uuid5(_SENSOR_NAMESPACE, composite_key)


def sensor_reading_to_data_point(reading: Reading) -> DataPoint:
    """Convierte SensorReading legacy a DataPoint canónico.
    
    Preserva sensor_id original en metadata para trazabilidad.
    
    Args:
        reading: Lectura de sensor (Reading o SensorReading)
    
    Returns:
        DataPoint canónico con metadata de trazabilidad
    """
    # Extraer sensor_id si existe (puede ser property deprecated)
    sensor_id = None
    try:
        # Intentar obtener sensor_id como int
        if hasattr(reading, 'sensor_id'):
            sensor_id = reading.sensor_id
        # Si no, intentar convertir series_id a int
        elif hasattr(reading, 'series_id'):
            try:
                sensor_id = int(reading.series_id)
            except (ValueError, TypeError):
                pass
    except Exception:
        pass
    
    # Construir metadata con trazabilidad
    metadata = {
        "legacy": True,
        "sensor_type": getattr(reading, 'sensor_type', ''),
    }
    
    if sensor_id is not None:
        metadata["sensor_id"] = sensor_id
    
    if hasattr(reading, 'device_id') and reading.device_id is not None:
        metadata["device_id"] = reading.device_id
    
    # Extraer quality si existe
    quality = getattr(reading, 'quality', 1.0)
    
    return DataPoint(
        value=float(reading.value),
        timestamp=float(reading.timestamp),
        quality=float(quality),
        metadata=metadata,
    )


def sensor_readings_to_time_window(
    readings: List[Reading],
    sensor_id: int,
    tenant_id: UUID,
) -> TimeWindow:
    """Convierte lista de SensorReading a TimeWindow canónico.
    
    Args:
        readings: Lista de lecturas de sensor
        sensor_id: ID numérico del sensor
        tenant_id: UUID del tenant
    
    Returns:
        TimeWindow con series_id UUID y puntos convertidos
    """
    # Convertir sensor_id a series_id UUID
    series_id = sensor_id_to_series_id(sensor_id, tenant_id)
    
    # Convertir cada reading a DataPoint
    points = tuple(sensor_reading_to_data_point(r) for r in readings)
    
    return TimeWindow(
        series_id=series_id,
        tenant_id=tenant_id,
        points=points,
    )
