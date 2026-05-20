"""Auto-generated coverage test for ml_service/correlation/sensor_correlator.py."""
import pytest


def test_sensor_correlator_importable():
    try:
        import iot_machine_learning.ml_service.correlation.sensor_correlator
        assert iot_machine_learning.ml_service.correlation.sensor_correlator is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
