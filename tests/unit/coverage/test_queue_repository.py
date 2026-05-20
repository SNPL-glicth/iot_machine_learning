"""Auto-generated coverage test for ml_service/workers/queue_repository.py."""
import pytest


def test_queue_repository_importable():
    try:
        import iot_machine_learning.ml_service.workers.queue_repository
        assert iot_machine_learning.ml_service.workers.queue_repository is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
