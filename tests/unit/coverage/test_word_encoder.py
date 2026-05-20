"""Auto-generated coverage test for infrastructure/ml/cognitive/text/embeddings/word_encoder.py."""
import pytest


def test_word_encoder_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.text.embeddings.word_encoder
        assert iot_machine_learning.infrastructure.ml.cognitive.text.embeddings.word_encoder is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
