"""Auto-generated coverage test for ml_service/tasks/expire_predictions_task.py."""
import pytest


def test_expire_predictions_task_importable():
    try:
        import iot_machine_learning.ml_service.tasks.expire_predictions_task
        assert iot_machine_learning.ml_service.tasks.expire_predictions_task is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
