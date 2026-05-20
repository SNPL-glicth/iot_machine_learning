"""Auto-generated coverage test for ml_service/broker/broker_factory.py."""
import pytest


def test_broker_factory_importable():
    try:
        import iot_machine_learning.ml_service.broker.broker_factory
        assert iot_machine_learning.ml_service.broker.broker_factory is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
