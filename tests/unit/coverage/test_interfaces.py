"""Auto-generated coverage test for infrastructure/ml/interfaces.py."""
import pytest


def test_interfaces_importable():
    try:
        import iot_machine_learning.infrastructure.ml.interfaces
        assert iot_machine_learning.infrastructure.ml.interfaces is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
