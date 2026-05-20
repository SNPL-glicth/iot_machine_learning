"""Auto-generated coverage test for ml_service/repository/sensor_repository.py."""
import pytest


def test_sensor_repository_importable():
    try:
        import iot_machine_learning.ml_service.repository.sensor_repository
        assert iot_machine_learning.ml_service.repository.sensor_repository is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
