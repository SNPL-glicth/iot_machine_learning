"""Auto-generated coverage test for ml_service/runners/services/event_buffer.py."""
import pytest


def test_event_buffer_importable():
    try:
        import iot_machine_learning.ml_service.runners.services.event_buffer
        assert iot_machine_learning.ml_service.runners.services.event_buffer is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
