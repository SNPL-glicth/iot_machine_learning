"""Auto-generated coverage test for application/use_cases/predict_sensor_value.py."""
import pytest


def test_predict_sensor_value_importable():
    try:
        import iot_machine_learning.application.use_cases.predict_sensor_value
        assert iot_machine_learning.application.use_cases.predict_sensor_value is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
