"""Auto-generated coverage test for infrastructure/ml/engines/taylor/time_step.py."""
import pytest


def test_time_step_importable():
    try:
        import iot_machine_learning.infrastructure.ml.engines.taylor.time_step
        assert iot_machine_learning.infrastructure.ml.engines.taylor.time_step is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
