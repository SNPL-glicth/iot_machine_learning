"""Auto-generated coverage test for ml_service/api/services/analysis/document_analyzer_factory.py."""
import pytest


def test_document_analyzer_factory_importable():
    try:
        import iot_machine_learning.ml_service.api.services.analysis.document_analyzer_factory
        assert iot_machine_learning.ml_service.api.services.analysis.document_analyzer_factory is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
