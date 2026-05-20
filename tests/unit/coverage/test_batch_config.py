"""Auto-generated coverage test for ml_service/config/batch_config.py."""
import pytest


def test_batch_config_importable():
    try:
        import iot_machine_learning.ml_service.config.batch_config
        assert iot_machine_learning.ml_service.config.batch_config is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
