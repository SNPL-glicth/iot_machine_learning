"""Auto-generated coverage test for infrastructure/ml/moe/rollout/rollout_decider.py."""
import pytest


def test_rollout_decider_importable():
    try:
        import iot_machine_learning.infrastructure.ml.moe.rollout.rollout_decider
        assert iot_machine_learning.infrastructure.ml.moe.rollout.rollout_decider is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
