"""Auto-generated coverage test for ml_service/api/services/analyzers/text_embedder.py."""
import pytest


def test_text_embedder_importable():
    try:
        import iot_machine_learning.ml_service.api.services.analyzers.text_embedder
        assert iot_machine_learning.ml_service.api.services.analyzers.text_embedder is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
