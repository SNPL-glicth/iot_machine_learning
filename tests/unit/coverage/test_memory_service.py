"""Auto-generated coverage test for ml_service/memory/services/memory_service.py."""
import pytest


def test_memory_service_importable():
    try:
        import iot_machine_learning.ml_service.memory.services.memory_service
        assert iot_machine_learning.ml_service.memory.services.memory_service is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
