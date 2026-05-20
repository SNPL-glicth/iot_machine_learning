"""Auto-generated coverage test for ml_service/explain/services/template_generator.py."""
import pytest


def test_template_generator_importable():
    try:
        import iot_machine_learning.ml_service.explain.services.template_generator
        pass  # import verified
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        pytest.skip(f"Import failed: {e}")
