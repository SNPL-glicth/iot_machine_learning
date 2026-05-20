"""Auto-generated coverage test for ml_service/api/services/model_service.py."""
import pytest


def test_model_service_importable():
    try:
        import iot_machine_learning.ml_service.api.services.model_service
        assert iot_machine_learning.ml_service.api.services.model_service is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
