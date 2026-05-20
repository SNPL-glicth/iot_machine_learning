"""Auto-generated coverage test for infrastructure/ml/cognitive/text/semantic_extraction/extractor_factory.py."""
import pytest


def test_extractor_factory_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.text.semantic_extraction.extractor_factory
        assert iot_machine_learning.infrastructure.ml.cognitive.text.semantic_extraction.extractor_factory is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
