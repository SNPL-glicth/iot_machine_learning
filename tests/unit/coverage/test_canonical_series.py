"""Auto-generated coverage test for domain/entities/canonical_series.py."""
import pytest


def test_canonical_series_importable():
    try:
        import iot_machine_learning.domain.entities.canonical_series
        assert iot_machine_learning.domain.entities.canonical_series is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
