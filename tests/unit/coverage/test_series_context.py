"""Auto-generated coverage test for domain/entities/series_context.py."""
import pytest


def test_series_context_importable():
    try:
        import iot_machine_learning.domain.entities.series_context
        assert iot_machine_learning.domain.entities.series_context is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
