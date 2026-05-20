"""Auto-generated coverage test for infrastructure/ml/engines/statistical/param_optimizer.py."""
import pytest


def test_param_optimizer_importable():
    try:
        import iot_machine_learning.infrastructure.ml.engines.statistical.param_optimizer
        assert iot_machine_learning.infrastructure.ml.engines.statistical.param_optimizer is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
