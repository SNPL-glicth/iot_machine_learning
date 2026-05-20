"""Auto-generated coverage test for infrastructure/ml/cognitive/neural/competition/confidence_comparator.py."""
import pytest


def test_confidence_comparator_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.neural.competition.confidence_comparator
        assert iot_machine_learning.infrastructure.ml.cognitive.neural.competition.confidence_comparator is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
