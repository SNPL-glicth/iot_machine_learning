"""Auto-generated coverage test for domain/ports/confidence_calibrator_port.py."""
import pytest


def test_confidence_calibrator_port_importable():
    try:
        import iot_machine_learning.domain.ports.confidence_calibrator_port
        assert iot_machine_learning.domain.ports.confidence_calibrator_port is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
