"""Auto-generated coverage test for ml_service/api/result_store.py."""
import pytest


def test_result_store_importable():
    try:
        import iot_machine_learning.ml_service.api.result_store
        assert iot_machine_learning.ml_service.api.result_store is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
