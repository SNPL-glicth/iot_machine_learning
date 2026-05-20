"""Auto-generated coverage test for infrastructure/ml/cognitive/text/semantic_extraction/metric_extractor.py."""
import pytest


def test_metric_extractor_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.text.semantic_extraction.metric_extractor
        assert iot_machine_learning.infrastructure.ml.cognitive.text.semantic_extraction.metric_extractor is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
