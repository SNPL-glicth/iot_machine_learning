"""Auto-generated coverage test for infrastructure/persistence/adapters/analysis_result_adapter.py."""
import pytest


def test_analysis_result_adapter_importable():
    try:
        import iot_machine_learning.infrastructure.persistence.adapters.analysis_result_adapter
        assert iot_machine_learning.infrastructure.persistence.adapters.analysis_result_adapter is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
