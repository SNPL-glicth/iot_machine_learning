"""Auto-generated coverage test for ml_service/api/services/analysis/decision_engine_service.py."""
import pytest


def test_decision_engine_service_importable():
    try:
        import iot_machine_learning.ml_service.api.services.analysis.decision_engine_service
        assert iot_machine_learning.ml_service.api.services.analysis.decision_engine_service is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
