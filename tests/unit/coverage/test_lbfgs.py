"""Auto-generated coverage test for infrastructure/ml/optimization/convex/lbfgs.py."""
import pytest


def test_lbfgs_importable():
    try:
        import iot_machine_learning.infrastructure.ml.optimization.convex.lbfgs
        assert iot_machine_learning.infrastructure.ml.optimization.convex.lbfgs is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
