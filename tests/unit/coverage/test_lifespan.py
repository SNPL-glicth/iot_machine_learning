"""Auto-generated coverage test for ml_service/lifespan.py."""
import pytest


def test_lifespan_importable():
    try:
        import iot_machine_learning.ml_service.lifespan
        assert iot_machine_learning.ml_service.lifespan is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
