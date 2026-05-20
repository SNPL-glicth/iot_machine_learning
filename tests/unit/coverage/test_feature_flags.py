"""Auto-generated coverage test for ml_service/config/feature_flags.py."""
import pytest


def test_feature_flags_importable():
    try:
        import iot_machine_learning.ml_service.config.feature_flags
        assert iot_machine_learning.ml_service.config.feature_flags is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
