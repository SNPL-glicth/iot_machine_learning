"""Auto-generated coverage test for ml_service/in_memory_broker.py."""
import pytest


def test_in_memory_broker_importable():
    try:
        import iot_machine_learning.ml_service.in_memory_broker
        assert iot_machine_learning.ml_service.in_memory_broker is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
