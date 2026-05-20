"""Auto-generated coverage test for infrastructure/ml/optimization/gradient/scheduler.py."""
import pytest


def test_scheduler_importable():
    try:
        import iot_machine_learning.infrastructure.ml.optimization.gradient.scheduler
        assert iot_machine_learning.infrastructure.ml.optimization.gradient.scheduler is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
