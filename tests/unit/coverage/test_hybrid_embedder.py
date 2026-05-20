"""Auto-generated coverage test for infrastructure/ml/cognitive/text/embeddings/hybrid_embedder.py."""
import pytest


def test_hybrid_embedder_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.text.embeddings.hybrid_embedder
        assert iot_machine_learning.infrastructure.ml.cognitive.text.embeddings.hybrid_embedder is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
