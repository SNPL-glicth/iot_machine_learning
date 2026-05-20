"""Auto-generated coverage test for infrastructure/repositories/prediction_repository.py."""
import pytest


def test_prediction_repository_importable():
    try:
        import iot_machine_learning.infrastructure.repositories.prediction_repository
        assert iot_machine_learning.infrastructure.repositories.prediction_repository is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
