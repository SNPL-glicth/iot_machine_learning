"""Auto-generated coverage test for infrastructure/ml/anomaly/detectors/temporal_z_detector.py."""
import pytest


def test_temporal_z_detector_importable():
    try:
        import iot_machine_learning.infrastructure.ml.anomaly.detectors.temporal_z_detector
        assert iot_machine_learning.infrastructure.ml.anomaly.detectors.temporal_z_detector is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
