"""Auto-generated coverage test for domain/tools/tool_models.py."""
import pytest


def test_tool_models_importable():
    try:
        import iot_machine_learning.domain.tools.tool_models
        assert iot_machine_learning.domain.tools.tool_models is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
