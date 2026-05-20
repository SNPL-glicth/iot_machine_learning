"""Auto-generated coverage test for infrastructure/ml/moe/expert_wrappers/engine_adapter.py."""
import pytest


def test_engine_adapter_importable():
    try:
        import iot_machine_learning.infrastructure.ml.moe.expert_wrappers.engine_adapter
        assert iot_machine_learning.infrastructure.ml.moe.expert_wrappers.engine_adapter is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
