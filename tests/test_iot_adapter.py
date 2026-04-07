"""Tests del adapter IoT → Zenin."""

from __future__ import annotations

import pytest
from uuid import UUID
from unittest.mock import MagicMock

from infrastructure.adapters.iot.sensor_adapter import (
    sensor_id_to_series_id,
    sensor_reading_to_data_point,
    sensor_readings_to_time_window,
)


class TestSensorIdToSeriesId:
    """Tests de conversión sensor_id → series_id UUID."""
    
    def test_deterministic(self):
        """Mismo input → mismo UUID siempre."""
        tenant = UUID("12345678-1234-5678-1234-567812345678")
        uuid1 = sensor_id_to_series_id(42, tenant)
        uuid2 = sensor_id_to_series_id(42, tenant)
        assert uuid1 == uuid2
    
    def test_different_sensors(self):
        """Sensores distintos → UUIDs distintos."""
        tenant = UUID("12345678-1234-5678-1234-567812345678")
        assert sensor_id_to_series_id(1, tenant) != sensor_id_to_series_id(2, tenant)
    
    def test_different_tenants(self):
        """Mismo sensor, tenant distinto → UUID distinto."""
        t1 = UUID("11111111-1111-1111-1111-111111111111")
        t2 = UUID("22222222-2222-2222-2222-222222222222")
        assert sensor_id_to_series_id(42, t1) != sensor_id_to_series_id(42, t2)
    
    def test_returns_uuid(self):
        """El resultado es un UUID válido."""
        tenant = UUID("12345678-1234-5678-1234-567812345678")
        result = sensor_id_to_series_id(1, tenant)
        assert isinstance(result, UUID)
    
    def test_idempotent_across_calls(self):
        """Múltiples llamadas con mismo input → mismo UUID."""
        tenant = UUID("12345678-1234-5678-1234-567812345678")
        results = [sensor_id_to_series_id(99, tenant) for _ in range(10)]
        assert len(set(results)) == 1


class TestSensorReadingToDataPoint:
    """Tests de conversión SensorReading → DataPoint."""
    
    def test_preserves_sensor_id_in_metadata(self):
        """sensor_id queda guardado en metadata para trazabilidad."""
        reading = MagicMock()
        reading.value = 23.5
        reading.timestamp = 1700000000.0
        reading.sensor_id = 99
        reading.sensor_type = "temperature"
        reading.device_id = None
        
        point = sensor_reading_to_data_point(reading)
        assert point.metadata["sensor_id"] == 99
        assert point.metadata["legacy"] is True
    
    def test_value_is_float(self):
        """El valor siempre es float."""
        reading = MagicMock()
        reading.value = 42  # int
        reading.timestamp = 1.0
        reading.sensor_id = 1
        reading.sensor_type = ""
        reading.device_id = None
        
        point = sensor_reading_to_data_point(reading)
        assert isinstance(point.value, float)
        assert point.value == 42.0
    
    def test_timestamp_is_float(self):
        """El timestamp siempre es float."""
        reading = MagicMock()
        reading.value = 10.0
        reading.timestamp = 1700000000  # int
        reading.sensor_id = 1
        reading.sensor_type = ""
        reading.device_id = None
        
        point = sensor_reading_to_data_point(reading)
        assert isinstance(point.timestamp, float)
    
    def test_default_quality(self):
        """Quality por defecto es 1.0."""
        reading = MagicMock()
        reading.value = 10.0
        reading.timestamp = 1.0
        reading.sensor_id = 1
        reading.sensor_type = ""
        reading.device_id = None
        
        point = sensor_reading_to_data_point(reading)
        assert point.quality == 1.0


class TestSensorReadingsToTimeWindow:
    """Tests de conversión List[SensorReading] → TimeWindow."""
    
    def test_builds_correct_window(self):
        """TimeWindow tiene series_id correcto y todos los puntos."""
        tenant = UUID("12345678-1234-5678-1234-567812345678")
        readings = []
        for i in range(5):
            r = MagicMock()
            r.value = float(i)
            r.timestamp = float(i)
            r.sensor_id = 7
            r.sensor_type = "temperature"
            r.device_id = None
            readings.append(r)
        
        window = sensor_readings_to_time_window(readings, sensor_id=7, tenant_id=tenant)
        assert window.series_id == sensor_id_to_series_id(7, tenant)
        assert window.tenant_id == tenant
        assert window.size == 5
        assert window.values == [0.0, 1.0, 2.0, 3.0, 4.0]
    
    def test_empty_readings(self):
        """Lista vacía produce ventana vacía."""
        tenant = UUID("12345678-1234-5678-1234-567812345678")
        window = sensor_readings_to_time_window([], sensor_id=1, tenant_id=tenant)
        assert window.is_empty
        assert window.size == 0
    
    def test_preserves_order(self):
        """Los puntos mantienen el orden de entrada."""
        tenant = UUID("12345678-1234-5678-1234-567812345678")
        readings = []
        for i in [5, 3, 1, 4, 2]:  # Desordenados
            r = MagicMock()
            r.value = float(i)
            r.timestamp = float(i)
            r.sensor_id = 1
            r.sensor_type = ""
            r.device_id = None
            readings.append(r)
        
        window = sensor_readings_to_time_window(readings, sensor_id=1, tenant_id=tenant)
        assert window.values == [5.0, 3.0, 1.0, 4.0, 2.0]
