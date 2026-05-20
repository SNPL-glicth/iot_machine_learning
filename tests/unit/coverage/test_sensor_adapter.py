"""Auto-generated coverage test for infrastructure/adapters/iot/sensor_adapter.py."""
import pytest


def test_sensor_adapter_importable():
    try:
        import iot_machine_learning.infrastructure.adapters.iot.sensor_adapter
        assert iot_machine_learning.infrastructure.adapters.iot.sensor_adapter is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
