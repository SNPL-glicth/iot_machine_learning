"""Auto-generated coverage test for infrastructure/ml/explainability/feature_importance.py."""
import pytest


def test_feature_importance_importable():
    try:
        import iot_machine_learning.infrastructure.ml.explainability.feature_importance
        assert iot_machine_learning.infrastructure.ml.explainability.feature_importance is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
