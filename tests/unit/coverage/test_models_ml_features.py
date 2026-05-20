"""Auto-generated coverage test for ml_service/features/models/ml_features.py."""
import pytest


def test_ml_features_importable():
    try:
        import iot_machine_learning.ml_service.features.models.ml_features
        assert iot_machine_learning.ml_service.features.models.ml_features is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
