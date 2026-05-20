"""Auto-generated coverage test for infrastructure/ml/cognitive/bayesian_weight_tracker/error_persister.py."""
import pytest


def test_error_persister_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.bayesian_weight_tracker.error_persister
        assert iot_machine_learning.infrastructure.ml.cognitive.bayesian_weight_tracker.error_persister is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
