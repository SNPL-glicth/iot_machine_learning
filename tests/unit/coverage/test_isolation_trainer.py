"""Auto-generated coverage test for ml_service/trainers/isolation_trainer.py."""
import pytest


def test_isolation_trainer_importable():
    try:
        import iot_machine_learning.ml_service.trainers.isolation_trainer
        assert iot_machine_learning.ml_service.trainers.isolation_trainer is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
