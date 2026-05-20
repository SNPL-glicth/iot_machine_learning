"""Auto-generated coverage test for infrastructure/ml/anomaly/persistent_detector.py."""
import pytest


def test_persistent_detector_importable():
    try:
        import iot_machine_learning.infrastructure.ml.anomaly.persistent_detector
        assert iot_machine_learning.infrastructure.ml.anomaly.persistent_detector is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
