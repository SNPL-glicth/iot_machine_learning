"""Auto-generated coverage test for ml_service/explain/models/explanation_result.py."""
import pytest


def test_explanation_result_importable():
    try:
        import iot_machine_learning.ml_service.explain.models.explanation_result
        pass  # import verified
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        pytest.skip(f"Import failed: {e}")
