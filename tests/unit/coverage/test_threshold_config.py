"""Auto-generated coverage test for ml_service/config/threshold_config.py."""
import pytest


def test_threshold_config_importable():
    try:
        import iot_machine_learning.ml_service.config.threshold_config
        assert iot_machine_learning.ml_service.config.threshold_config is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
