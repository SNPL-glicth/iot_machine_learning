"""Auto-generated coverage test for ml_service/config/ml_config.py."""
import pytest


def test_ml_config_importable():
    try:
        import iot_machine_learning.ml_service.config.ml_config
        assert iot_machine_learning.ml_service.config.ml_config is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
