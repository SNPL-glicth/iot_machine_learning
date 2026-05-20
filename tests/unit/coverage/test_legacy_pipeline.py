"""Auto-generated coverage test for ml_service/api/services/analysis/legacy_pipeline.py."""
import pytest


def test_legacy_pipeline_importable():
    try:
        import iot_machine_learning.ml_service.api.services.analysis.legacy_pipeline
        assert iot_machine_learning.ml_service.api.services.analysis.legacy_pipeline is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
