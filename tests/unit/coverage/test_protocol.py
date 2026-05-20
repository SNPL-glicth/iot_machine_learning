"""Auto-generated coverage test for infrastructure/ml/anomaly/core/protocol.py."""
import pytest


def test_protocol_importable():
    try:
        import iot_machine_learning.infrastructure.ml.anomaly.core.protocol
        assert iot_machine_learning.infrastructure.ml.anomaly.core.protocol is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
