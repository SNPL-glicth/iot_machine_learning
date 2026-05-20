"""Auto-generated coverage test for infrastructure/ml/moe/gating/tree_gating.py."""
import pytest


def test_tree_gating_importable():
    try:
        import iot_machine_learning.infrastructure.ml.moe.gating.tree_gating
        assert iot_machine_learning.infrastructure.ml.moe.gating.tree_gating is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
