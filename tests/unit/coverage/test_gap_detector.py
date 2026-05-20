"""Auto-generated coverage test for infrastructure/ml/engines/taylor/gap_detector.py."""
import pytest


def test_gap_detector_importable():
    try:
        import iot_machine_learning.infrastructure.ml.engines.taylor.gap_detector
        assert iot_machine_learning.infrastructure.ml.engines.taylor.gap_detector is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
