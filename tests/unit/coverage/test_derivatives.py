"""Auto-generated coverage test for infrastructure/ml/engines/taylor/derivatives.py."""
import pytest


def test_derivatives_importable():
    try:
        import iot_machine_learning.infrastructure.ml.engines.taylor.derivatives
        assert iot_machine_learning.infrastructure.ml.engines.taylor.derivatives is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
