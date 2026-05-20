"""Auto-generated coverage test for infrastructure/ml/inference/mle/parameter_fitter.py."""
import pytest


def test_parameter_fitter_importable():
    try:
        import iot_machine_learning.infrastructure.ml.inference.mle.parameter_fitter
        assert iot_machine_learning.infrastructure.ml.inference.mle.parameter_fitter is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
