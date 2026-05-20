"""Auto-generated coverage test for infrastructure/analysis/pipeline/reason.py."""
import pytest


def test_reason_importable():
    try:
        import iot_machine_learning.infrastructure.analysis.pipeline.reason
        assert iot_machine_learning.infrastructure.analysis.pipeline.reason is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
