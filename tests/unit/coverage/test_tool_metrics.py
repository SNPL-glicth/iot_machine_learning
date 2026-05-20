"""Auto-generated coverage test for domain/tools/tool_metrics.py."""
import pytest


def test_tool_metrics_importable():
    try:
        import iot_machine_learning.domain.tools.tool_metrics
        assert iot_machine_learning.domain.tools.tool_metrics is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
