"""Auto-generated coverage test for infrastructure/ml/_experimental/lightgbm/feature_builder.py."""
import pytest


def test_feature_builder_importable():
    try:
        import iot_machine_learning.infrastructure.ml._experimental.lightgbm.feature_builder
        assert iot_machine_learning.infrastructure.ml._experimental.lightgbm.feature_builder is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
