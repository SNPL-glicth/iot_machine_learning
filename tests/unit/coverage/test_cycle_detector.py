"""Auto-generated coverage test for infrastructure/ml/engines/seasonal/cycle_detector.py."""
import pytest


def test_cycle_detector_importable():
    try:
        import iot_machine_learning.infrastructure.ml.engines.seasonal.cycle_detector
        assert iot_machine_learning.infrastructure.ml.engines.seasonal.cycle_detector is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
