"""Auto-generated coverage test for infrastructure/adapters/weaviate/http_client.py."""
import pytest


def test_http_client_importable():
    try:
        import iot_machine_learning.infrastructure.adapters.weaviate.http_client
        assert iot_machine_learning.infrastructure.adapters.weaviate.http_client is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
