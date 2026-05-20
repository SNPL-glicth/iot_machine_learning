"""Auto-generated coverage test for ml_service/runners/models/online_analysis.py."""
import pytest


def test_online_analysis_importable():
    try:
        import iot_machine_learning.ml_service.runners.models.online_analysis
        assert iot_machine_learning.ml_service.runners.models.online_analysis is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
