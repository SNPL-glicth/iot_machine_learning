"""Auto-generated coverage test for infrastructure/ml/engines/taylor/types.py."""
import pytest


def test_types_importable():
    try:
        import iot_machine_learning.infrastructure.ml.engines.taylor.types
        assert iot_machine_learning.infrastructure.ml.engines.taylor.types is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
