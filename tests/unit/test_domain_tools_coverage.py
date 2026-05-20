"""Coverage tests for domain/tools/ — tool infrastructure.

Covers: tool_executor, tool_registry, tool_models, tool_metrics,
tool_base, tool_guard, tool, iot_tools
"""
from __future__ import annotations

import pytest


class TestToolModules:
    def test_tool_executor_import(self):
        from iot_machine_learning.domain.tools import tool_executor
        assert tool_executor is not None

    def test_tool_registry_import(self):
        from iot_machine_learning.domain.tools import tool_registry
        assert tool_registry is not None

    def test_tool_models_import(self):
        from iot_machine_learning.domain.tools import tool_models
        assert tool_models is not None

    def test_tool_metrics_import(self):
        from iot_machine_learning.domain.tools import tool_metrics
        assert tool_metrics is not None

    def test_tool_base_import(self):
        from iot_machine_learning.domain.tools import tool_base
        assert tool_base is not None

    def test_tool_guard_import(self):
        from iot_machine_learning.domain.tools import tool_guard
        assert tool_guard is not None

    def test_tool_import(self):
        from iot_machine_learning.domain.tools import tool
        assert tool is not None

    def test_iot_tools_import(self):
        from iot_machine_learning.domain.tools import iot_tools
        assert iot_tools is not None
