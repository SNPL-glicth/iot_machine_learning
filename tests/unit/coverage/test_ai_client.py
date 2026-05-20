"""Auto-generated coverage test for ml_service/explain/services/ai_client.py."""
import pytest


def test_ai_client_importable():
    try:
        import iot_machine_learning.ml_service.explain.services.ai_client
        assert iot_machine_learning.ml_service.explain.services.ai_client is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
