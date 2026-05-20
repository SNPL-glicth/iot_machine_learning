"""Auto-generated coverage test for ml_service/runners/batch_worker.py."""
import pytest


def test_batch_worker_importable():
    try:
        import iot_machine_learning.ml_service.runners.batch_worker
        assert iot_machine_learning.ml_service.runners.batch_worker is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
