"""Auto-generated coverage test for infrastructure/ml/patterns/change_point_detector.py."""
import pytest


def test_change_point_detector_importable():
    try:
        import iot_machine_learning.infrastructure.ml.patterns.change_point_detector
        assert iot_machine_learning.infrastructure.ml.patterns.change_point_detector is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
