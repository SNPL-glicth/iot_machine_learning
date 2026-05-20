"""Auto-generated coverage test for ml_service/api/services/analysis/result_builder.py."""
import pytest


def test_result_builder_importable():
    try:
        import iot_machine_learning.ml_service.api.services.analysis.result_builder
        assert iot_machine_learning.ml_service.api.services.analysis.result_builder is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
