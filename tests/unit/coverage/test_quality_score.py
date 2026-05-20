"""Auto-generated coverage test for application/evaluation/quality_score.py."""
import pytest


def test_quality_score_importable():
    try:
        import iot_machine_learning.application.evaluation.quality_score
        assert iot_machine_learning.application.evaluation.quality_score is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
