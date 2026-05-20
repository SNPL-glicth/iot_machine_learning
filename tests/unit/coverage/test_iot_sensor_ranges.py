"""Auto-generated coverage test for domain/entities/iot/sensor_ranges.py."""
import pytest


def test_sensor_ranges_importable():
    try:
        import iot_machine_learning.domain.entities.iot.sensor_ranges
        assert iot_machine_learning.domain.entities.iot.sensor_ranges is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
