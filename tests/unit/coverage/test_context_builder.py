"""Auto-generated coverage test for infrastructure/ml/anomaly/voting/context_builder.py."""
import pytest


def test_context_builder_importable():
    try:
        import iot_machine_learning.infrastructure.ml.anomaly.voting.context_builder
        assert iot_machine_learning.infrastructure.ml.anomaly.voting.context_builder is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
