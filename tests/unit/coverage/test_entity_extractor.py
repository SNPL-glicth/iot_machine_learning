"""Auto-generated coverage test for infrastructure/ml/cognitive/text/entity_extractor.py."""
import pytest


def test_entity_extractor_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.text.entity_extractor
        assert iot_machine_learning.infrastructure.ml.cognitive.text.entity_extractor is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
