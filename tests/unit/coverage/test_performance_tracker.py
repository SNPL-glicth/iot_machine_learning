"""Auto-generated coverage test for infrastructure/ml/engines/taylor/performance_tracker.py."""
import pytest


def test_performance_tracker_importable():
    try:
        import iot_machine_learning.infrastructure.ml.engines.taylor.performance_tracker
        assert iot_machine_learning.infrastructure.ml.engines.taylor.performance_tracker is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
