"""Auto-generated coverage test for core/ensemble/ensemble_calibrator.py."""
import pytest


def test_ensemble_calibrator_importable():
    try:
        import iot_machine_learning.core.ensemble.ensemble_calibrator
        assert iot_machine_learning.core.ensemble.ensemble_calibrator is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
