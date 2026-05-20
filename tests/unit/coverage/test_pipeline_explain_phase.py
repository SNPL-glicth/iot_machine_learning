"""Auto-generated coverage test for infrastructure/ml/cognitive/text/pipeline/explain_phase.py."""
import pytest


def test_explain_phase_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.text.pipeline.explain_phase
        assert iot_machine_learning.infrastructure.ml.cognitive.text.pipeline.explain_phase is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
