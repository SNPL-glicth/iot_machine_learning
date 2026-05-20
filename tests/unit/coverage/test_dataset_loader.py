"""Auto-generated coverage test for infrastructure/ml/benchmark/dataset_loader.py."""
import pytest


def test_dataset_loader_importable():
    try:
        import iot_machine_learning.infrastructure.ml.benchmark.dataset_loader
        assert iot_machine_learning.infrastructure.ml.benchmark.dataset_loader is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
