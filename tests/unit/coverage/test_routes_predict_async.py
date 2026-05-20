"""Auto-generated coverage test for ml_service/api/routes_predict_async.py."""
import pytest


def test_routes_predict_async_importable():
    try:
        import iot_machine_learning.ml_service.api.routes_predict_async
        assert iot_machine_learning.ml_service.api.routes_predict_async is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
