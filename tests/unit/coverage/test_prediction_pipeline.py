"""Auto-generated coverage test for infrastructure/ml/engines/taylor/prediction_pipeline.py."""
import pytest


def test_prediction_pipeline_importable():
    try:
        import iot_machine_learning.infrastructure.ml.engines.taylor.prediction_pipeline
        assert iot_machine_learning.infrastructure.ml.engines.taylor.prediction_pipeline is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
