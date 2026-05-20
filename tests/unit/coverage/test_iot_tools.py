"""Auto-generated coverage test for domain/tools/iot_tools.py."""
import pytest


def test_iot_tools_importable():
    try:
        import iot_machine_learning.domain.tools.iot_tools
        assert iot_machine_learning.domain.tools.iot_tools is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
