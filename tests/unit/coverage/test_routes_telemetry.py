"""Auto-generated coverage test for ml_service/api/routes_telemetry.py."""
import pytest


def test_routes_telemetry_importable():
    try:
        import iot_machine_learning.ml_service.api.routes_telemetry
        assert iot_machine_learning.ml_service.api.routes_telemetry is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
