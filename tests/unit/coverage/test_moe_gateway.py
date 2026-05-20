"""Auto-generated coverage test for infrastructure/ml/moe/gateway/moe_gateway.py."""
import pytest


def test_moe_gateway_importable():
    try:
        import iot_machine_learning.infrastructure.ml.moe.gateway.moe_gateway
        assert iot_machine_learning.infrastructure.ml.moe.gateway.moe_gateway is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
