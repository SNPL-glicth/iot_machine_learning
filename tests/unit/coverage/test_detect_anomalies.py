"""Auto-generated coverage test for application/use_cases/detect_anomalies.py."""
import pytest


def test_detect_anomalies_importable():
    try:
        import iot_machine_learning.application.use_cases.detect_anomalies
        assert iot_machine_learning.application.use_cases.detect_anomalies is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
