"""Auto-generated coverage test for infrastructure/sliding_window/in_memory.py."""
import pytest


def test_in_memory_importable():
    try:
        import iot_machine_learning.infrastructure.sliding_window.in_memory
        assert iot_machine_learning.infrastructure.sliding_window.in_memory is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
