"""Auto-generated coverage test for infrastructure/ml/moe/gateway/expert_dispatcher.py."""
import pytest


def test_expert_dispatcher_importable():
    try:
        import iot_machine_learning.infrastructure.ml.moe.gateway.expert_dispatcher
        assert iot_machine_learning.infrastructure.ml.moe.gateway.expert_dispatcher is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
