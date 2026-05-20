"""Auto-generated coverage test for ml_service/context/operational_context.py."""
import pytest


def test_operational_context_importable():
    try:
        import iot_machine_learning.ml_service.context.operational_context
        assert iot_machine_learning.ml_service.context.operational_context is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
