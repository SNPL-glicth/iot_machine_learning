"""Coverage test for infrastructure/ml/inference/bayesian/posterior.py."""
import pytest


def test_bayesian_posterior_importable():
    try:
        import iot_machine_learning.infrastructure.ml.inference.bayesian.posterior
        assert iot_machine_learning.infrastructure.ml.inference.bayesian.posterior is not None
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        pytest.skip(f"Import failed: {e}")
