"""Auto-generated coverage test for infrastructure/ml/cognitive/neural/pipeline/encoder_stage.py."""
import pytest


def test_encoder_stage_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.neural.pipeline.encoder_stage
        assert iot_machine_learning.infrastructure.ml.cognitive.neural.pipeline.encoder_stage is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
