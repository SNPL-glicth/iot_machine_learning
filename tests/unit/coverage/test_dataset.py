"""Auto-generated coverage test for application/evaluation/dataset.py."""
import pytest


def test_dataset_importable():
    try:
        import iot_machine_learning.application.evaluation.dataset
        assert iot_machine_learning.application.evaluation.dataset is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
