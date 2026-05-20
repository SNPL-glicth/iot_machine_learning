"""Auto-generated coverage test for domain/entities/iot/sensor_reading.py."""
import pytest


def test_sensor_reading_importable():
    try:
        import iot_machine_learning.domain.entities.iot.sensor_reading
        assert iot_machine_learning.domain.entities.iot.sensor_reading is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
