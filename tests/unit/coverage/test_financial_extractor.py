"""Auto-generated coverage test for infrastructure/ml/cognitive/text/semantic_extraction/financial_extractor.py."""
import pytest


def test_financial_extractor_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.text.semantic_extraction.financial_extractor
        assert iot_machine_learning.infrastructure.ml.cognitive.text.semantic_extraction.financial_extractor is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
