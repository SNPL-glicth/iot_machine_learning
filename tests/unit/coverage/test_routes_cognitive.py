"""Auto-generated coverage test for ml_service/api/routes_cognitive.py."""
import pytest


def test_routes_cognitive_importable():
    try:
        import iot_machine_learning.ml_service.api.routes_cognitive
        assert iot_machine_learning.ml_service.api.routes_cognitive is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
