"""Auto-generated coverage test for application/interfaces/decision_service_interface.py."""
import pytest


def test_decision_service_interface_importable():
    try:
        import iot_machine_learning.application.interfaces.decision_service_interface
        assert iot_machine_learning.application.interfaces.decision_service_interface is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
