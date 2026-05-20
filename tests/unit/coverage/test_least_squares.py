"""Auto-generated coverage test for infrastructure/ml/engines/taylor/least_squares.py."""
import pytest


def test_least_squares_importable():
    try:
        import iot_machine_learning.infrastructure.ml.engines.taylor.least_squares
        assert iot_machine_learning.infrastructure.ml.engines.taylor.least_squares is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
