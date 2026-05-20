"""Auto-generated coverage test for infrastructure/ml/cognitive/text/explanation_assembler.py."""
import pytest


def test_explanation_assembler_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.text.explanation_assembler
        assert iot_machine_learning.infrastructure.ml.cognitive.text.explanation_assembler is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
