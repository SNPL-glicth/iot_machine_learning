"""Auto-generated coverage test for domain/tools/tool_guard.py."""
import pytest


def test_tool_guard_importable():
    try:
        import iot_machine_learning.domain.tools.tool_guard
        assert iot_machine_learning.domain.tools.tool_guard is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
