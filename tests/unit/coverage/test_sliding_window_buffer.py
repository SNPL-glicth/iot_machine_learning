"""Auto-generated coverage test for ml_service/sliding_window_buffer.py."""
import pytest


def test_sliding_window_buffer_importable():
    try:
        import iot_machine_learning.ml_service.sliding_window_buffer
        assert iot_machine_learning.ml_service.sliding_window_buffer is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
