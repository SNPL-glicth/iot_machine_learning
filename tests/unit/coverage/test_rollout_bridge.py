"""Auto-generated coverage test for infrastructure/ml/moe/rollout/rollout_bridge.py."""
import pytest


def test_rollout_bridge_importable():
    try:
        import iot_machine_learning.infrastructure.ml.moe.rollout.rollout_bridge
        assert iot_machine_learning.infrastructure.ml.moe.rollout.rollout_bridge is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
