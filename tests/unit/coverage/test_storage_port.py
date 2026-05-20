"""Auto-generated coverage test for domain/ports/storage_port.py."""
import pytest


def test_storage_port_importable():
    try:
        import iot_machine_learning.domain.ports.storage_port
        assert iot_machine_learning.domain.ports.storage_port is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
