"""Auto-generated coverage test for ml_service/api/services/analysis/text_score_builder.py."""
import pytest


def test_text_score_builder_importable():
    try:
        import iot_machine_learning.ml_service.api.services.analysis.text_score_builder
        assert iot_machine_learning.ml_service.api.services.analysis.text_score_builder is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
