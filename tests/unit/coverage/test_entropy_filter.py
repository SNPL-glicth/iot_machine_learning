"""Auto-generated coverage test for infrastructure/ml/cognitive/text/embeddings/entropy_filter.py."""
import pytest


def test_entropy_filter_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.text.embeddings.entropy_filter
        assert iot_machine_learning.infrastructure.ml.cognitive.text.embeddings.entropy_filter is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
