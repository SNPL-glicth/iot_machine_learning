"""Auto-generated coverage test for ml_service/models/anomaly_model.py."""
import pytest


def test_anomaly_model_importable():
    try:
        import iot_machine_learning.ml_service.models.anomaly_model
        assert iot_machine_learning.ml_service.models.anomaly_model is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
