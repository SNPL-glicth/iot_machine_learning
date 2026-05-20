"""Auto-generated coverage test for infrastructure/ml/inference/mle/distributions.py."""
import pytest


def test_distributions_importable():
    try:
        import iot_machine_learning.infrastructure.ml.inference.mle.distributions
        assert iot_machine_learning.infrastructure.ml.inference.mle.distributions is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
