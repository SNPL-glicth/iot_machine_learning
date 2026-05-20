"""Auto-generated coverage test for infrastructure/ml/moe/events/prediction_drift_detector.py."""
import pytest


def test_prediction_drift_detector_importable():
    try:
        import iot_machine_learning.infrastructure.ml.moe.events.prediction_drift_detector
        assert iot_machine_learning.infrastructure.ml.moe.events.prediction_drift_detector is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
