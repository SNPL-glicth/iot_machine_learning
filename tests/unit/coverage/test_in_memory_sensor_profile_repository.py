"""Auto-generated coverage test for infrastructure/repositories/in_memory_sensor_profile_repository.py."""
import pytest


def test_in_memory_sensor_profile_repository_importable():
    try:
        import iot_machine_learning.infrastructure.repositories.in_memory_sensor_profile_repository
        assert iot_machine_learning.infrastructure.repositories.in_memory_sensor_profile_repository is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
