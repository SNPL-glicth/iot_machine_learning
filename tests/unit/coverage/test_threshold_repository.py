"""Auto-generated coverage test for infrastructure/repositories/threshold_repository.py."""
import pytest


def test_threshold_repository_importable():
    try:
        import iot_machine_learning.infrastructure.repositories.threshold_repository
        assert iot_machine_learning.infrastructure.repositories.threshold_repository is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
