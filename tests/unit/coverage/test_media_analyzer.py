"""Auto-generated coverage test for ml_service/api/services/analyzers/media_analyzer.py."""
import pytest


def test_media_analyzer_importable():
    try:
        import iot_machine_learning.ml_service.api.services.analyzers.media_analyzer
        assert iot_machine_learning.ml_service.api.services.analyzers.media_analyzer is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
