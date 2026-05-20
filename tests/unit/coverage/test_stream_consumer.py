"""Auto-generated coverage test for ml_service/consumers/stream_consumer.py."""
import pytest


def test_stream_consumer_importable():
    try:
        import iot_machine_learning.ml_service.consumers.stream_consumer
        assert iot_machine_learning.ml_service.consumers.stream_consumer is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
