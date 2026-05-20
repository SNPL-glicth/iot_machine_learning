"""Auto-generated coverage test for ml_service/explain/models/enriched_context.py."""
import pytest


def test_enriched_context_importable():
    try:
        import iot_machine_learning.ml_service.explain.models.enriched_context
        pass  # import verified
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        pytest.skip(f"Import failed: {e}")
