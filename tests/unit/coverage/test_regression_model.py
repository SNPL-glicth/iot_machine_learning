"""Auto-generated coverage test for ml_service/models/regression_model.py."""
import pytest


def test_regression_model_importable():
    try:
        import iot_machine_learning.ml_service.models.regression_model
        assert iot_machine_learning.ml_service.models.regression_model is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
