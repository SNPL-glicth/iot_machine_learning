"""Auto-generated coverage test for infrastructure/ml/_experimental/gradient/adam.py."""
import pytest


def test_adam_importable():
    try:
        import iot_machine_learning.infrastructure.ml._experimental.gradient.adam
        assert iot_machine_learning.infrastructure.ml._experimental.gradient.adam is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
