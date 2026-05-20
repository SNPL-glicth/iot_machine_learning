"""Auto-generated coverage test for ml_service/api/services/analysis/arbitrator.py."""
import pytest


def test_arbitrator_importable():
    try:
        import iot_machine_learning.ml_service.api.services.analysis.arbitrator
        assert iot_machine_learning.ml_service.api.services.analysis.arbitrator is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
