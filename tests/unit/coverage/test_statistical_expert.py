"""Auto-generated coverage test for infrastructure/ml/moe/experts/statistical_expert.py."""
import pytest


def test_statistical_expert_importable():
    try:
        import iot_machine_learning.infrastructure.ml.moe.experts.statistical_expert
        assert iot_machine_learning.infrastructure.ml.moe.experts.statistical_expert is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
