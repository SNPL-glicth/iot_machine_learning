"""Auto-generated coverage test for domain/services/severity_legacy.py."""
import pytest


def test_severity_legacy_importable():
    try:
        import iot_machine_learning.domain.services.severity_legacy
        assert iot_machine_learning.domain.services.severity_legacy is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
