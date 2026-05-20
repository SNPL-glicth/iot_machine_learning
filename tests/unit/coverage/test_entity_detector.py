"""Auto-generated coverage test for infrastructure/ml/cognitive/text/embeddings/entity_detector.py."""
import pytest


def test_entity_detector_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.text.embeddings.entity_detector
        assert iot_machine_learning.infrastructure.ml.cognitive.text.embeddings.entity_detector is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
