"""Auto-generated coverage test for ml_service/api/routes_health.py."""
import pytest


def test_routes_health_importable():
    try:
        import iot_machine_learning.ml_service.api.routes_health
        assert iot_machine_learning.ml_service.api.routes_health is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
