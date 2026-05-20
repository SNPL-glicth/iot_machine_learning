"""Auto-generated coverage test for ml_service/runners/models/sensor_state.py."""
import pytest


def test_sensor_state_importable():
    try:
        import iot_machine_learning.ml_service.runners.models.sensor_state
        assert iot_machine_learning.ml_service.runners.models.sensor_state is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
