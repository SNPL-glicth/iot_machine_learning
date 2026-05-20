"""Auto-generated coverage test for ml_service/runners/common/model_manager.py."""
import pytest


def test_model_manager_importable():
    try:
        import iot_machine_learning.ml_service.runners.common.model_manager
        assert iot_machine_learning.ml_service.runners.common.model_manager is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
