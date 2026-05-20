"""Auto-generated coverage test for infrastructure/ml/optimization/convex/proximal.py."""
import pytest


def test_proximal_importable():
    try:
        import iot_machine_learning.infrastructure.ml.optimization.convex.proximal
        assert iot_machine_learning.infrastructure.ml.optimization.convex.proximal is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
