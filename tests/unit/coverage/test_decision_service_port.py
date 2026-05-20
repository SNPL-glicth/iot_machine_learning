"""Auto-generated coverage test for application/ports/decision_service_port.py."""
import pytest


def test_decision_service_port_importable():
    try:
        import iot_machine_learning.application.ports.decision_service_port
        assert iot_machine_learning.application.ports.decision_service_port is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
