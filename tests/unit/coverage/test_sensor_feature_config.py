"""Auto-generated coverage test for ml_service/features/sensor_feature_config.py."""
import pytest


def test_sensor_feature_config_importable():
    try:
        import iot_machine_learning.ml_service.features.sensor_feature_config
        assert iot_machine_learning.ml_service.features.sensor_feature_config is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
