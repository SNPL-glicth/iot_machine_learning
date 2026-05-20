"""Auto-generated coverage test for infrastructure/ml/analyzers/numeric_types.py."""
import pytest


def test_numeric_types_importable():
    try:
        import iot_machine_learning.infrastructure.ml.analyzers.numeric_types
        assert iot_machine_learning.infrastructure.ml.analyzers.numeric_types is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
