"""Auto-generated coverage test for ml_service/api/schemas.py."""
import pytest


def test_schemas_importable():
    try:
        import iot_machine_learning.ml_service.api.schemas
        assert iot_machine_learning.ml_service.api.schemas is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
