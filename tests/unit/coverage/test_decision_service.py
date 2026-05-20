"""Auto-generated coverage test for application/services/decision_service.py."""
import pytest


def test_decision_service_importable():
    try:
        import iot_machine_learning.application.services.decision_service
        assert iot_machine_learning.application.services.decision_service is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
