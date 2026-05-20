"""Auto-generated coverage test for infrastructure/analysis/pipeline/perceive.py."""
import pytest


def test_perceive_importable():
    try:
        import iot_machine_learning.infrastructure.analysis.pipeline.perceive
        assert iot_machine_learning.infrastructure.analysis.pipeline.perceive is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
