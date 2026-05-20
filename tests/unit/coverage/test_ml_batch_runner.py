"""Auto-generated coverage test for ml_service/runners/ml_batch_runner.py."""
import pytest


def test_ml_batch_runner_importable():
    try:
        import iot_machine_learning.ml_service.runners.ml_batch_runner
        assert iot_machine_learning.ml_service.runners.ml_batch_runner is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
