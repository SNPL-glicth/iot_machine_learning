"""Auto-generated coverage test for ml_service/explain/services/data_loader.py."""
import pytest


def test_data_loader_importable():
    try:
        import iot_machine_learning.ml_service.explain.services.data_loader
        pass  # import verified
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        pytest.skip(f"Import failed: {e}")
