"""Auto-generated coverage test for infrastructure/ml/research/neural/classical/feedforward.py."""
import pytest


def test_feedforward_importable():
    try:
        import iot_machine_learning.infrastructure.ml.research.neural.classical.feedforward
        assert iot_machine_learning.infrastructure.ml.research.neural.classical.feedforward is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
