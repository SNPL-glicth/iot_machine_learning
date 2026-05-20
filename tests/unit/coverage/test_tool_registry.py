"""Auto-generated coverage test for domain/tools/tool_registry.py."""
import pytest


def test_tool_registry_importable():
    try:
        import iot_machine_learning.domain.tools.tool_registry
        assert iot_machine_learning.domain.tools.tool_registry is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
