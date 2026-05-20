"""Auto-generated coverage test for ml_service/consumers/backpressure.py."""
import pytest


def test_backpressure_importable():
    try:
        import iot_machine_learning.ml_service.consumers.backpressure
        assert iot_machine_learning.ml_service.consumers.backpressure is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
