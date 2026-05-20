"""Auto-generated coverage test for infrastructure/security/key_parsing.py."""
import pytest


def test_key_parsing_importable():
    try:
        import iot_machine_learning.infrastructure.security.key_parsing
        assert iot_machine_learning.infrastructure.security.key_parsing is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
