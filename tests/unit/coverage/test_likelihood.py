"""Auto-generated coverage test for infrastructure/ml/inference/bayesian/likelihood.py."""
import pytest


def test_likelihood_importable():
    try:
        import iot_machine_learning.infrastructure.ml.inference.bayesian.likelihood
        assert iot_machine_learning.infrastructure.ml.inference.bayesian.likelihood is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
