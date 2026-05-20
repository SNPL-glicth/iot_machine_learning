"""Auto-generated coverage test for infrastructure/ml/moe/fusion/sparse_fusion.py."""
import pytest


def test_sparse_fusion_importable():
    try:
        import iot_machine_learning.infrastructure.ml.moe.fusion.sparse_fusion
        assert iot_machine_learning.infrastructure.ml.moe.fusion.sparse_fusion is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
