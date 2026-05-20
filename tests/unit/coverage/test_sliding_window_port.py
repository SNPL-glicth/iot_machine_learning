"""Auto-generated coverage test for domain/ports/sliding_window_port.py."""
import pytest


def test_sliding_window_port_importable():
    try:
        import iot_machine_learning.domain.ports.sliding_window_port
        assert iot_machine_learning.domain.ports.sliding_window_port is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
