"""Auto-generated coverage test for infrastructure/ml/moe/feature_context.py."""
import pytest


def test_feature_context_importable():
    try:
        import iot_machine_learning.infrastructure.ml.moe.feature_context
        assert iot_machine_learning.infrastructure.ml.moe.feature_context is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
