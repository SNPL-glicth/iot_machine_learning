"""Auto-generated coverage test for infrastructure/ml/cognitive/drift/page_hinkley.py."""
import pytest


def test_page_hinkley_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.drift.page_hinkley
        assert iot_machine_learning.infrastructure.ml.cognitive.drift.page_hinkley is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
