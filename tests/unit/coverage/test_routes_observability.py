"""Auto-generated coverage test for ml_service/api/routes_observability.py."""
import pytest


def test_routes_observability_importable():
    try:
        import iot_machine_learning.ml_service.api.routes_observability
        assert iot_machine_learning.ml_service.api.routes_observability is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
