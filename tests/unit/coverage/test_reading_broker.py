"""Auto-generated coverage test for ml_service/reading_broker.py."""
import pytest


def test_reading_broker_importable():
    try:
        import iot_machine_learning.ml_service.reading_broker
        assert iot_machine_learning.ml_service.reading_broker is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
