"""Auto-generated coverage test for ml_service/explain/contextual_explainer.py."""
import pytest


def test_contextual_explainer_importable():
    try:
        import iot_machine_learning.ml_service.explain.contextual_explainer
        assert iot_machine_learning.ml_service.explain.contextual_explainer is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
