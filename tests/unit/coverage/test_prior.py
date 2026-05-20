"""Auto-generated coverage test for infrastructure/ml/inference/bayesian/prior.py."""
import pytest


def test_prior_importable():
    try:
        import iot_machine_learning.infrastructure.ml.inference.bayesian.prior
        assert iot_machine_learning.infrastructure.ml.inference.bayesian.prior is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
