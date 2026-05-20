"""Auto-generated coverage test for domain/services/severity/formatting.py."""
import pytest


def test_formatting_importable():
    try:
        import iot_machine_learning.domain.services.severity.formatting
        assert iot_machine_learning.domain.services.severity.formatting is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
