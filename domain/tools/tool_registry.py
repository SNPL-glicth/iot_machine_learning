"""Tool registry for managing tools.

Refactored to use split components:
- tool_executor.py: execution logic
- tool_metrics.py: metrics tracking
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .tool import Tool, ToolContext, ToolResult
from .tool_executor import ToolExecutor
from .tool_metrics import ToolMetrics

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing available tools.
    
    Delegates execution to ToolExecutor and metrics to ToolMetrics.
    """
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._metrics = ToolMetrics()
        self._executor = ToolExecutor(self._metrics)
    
    def register(self, tool: Tool) -> None:
        """Register a tool in the registry."""
        name = tool.name
        if name in self._tools:
            logger.warning(f"Tool '{name}' already registered, overwriting")
        
        self._tools[name] = tool
        logger.info(f"Tool registered: {name} v{tool.version}")
    
    def unregister(self, tool_name: str) -> bool:
        """Unregister a tool."""
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.info(f"Tool unregistered: {tool_name}")
            return True
        return False
    
    def get(self, tool_name: str) -> Optional[Tool]:
        """Get tool by name."""
        return self._tools.get(tool_name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools with metadata."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "version": tool.version,
            }
            for tool in self._tools.values()
        ]
    
    def guard_and_execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: ToolContext,
        auto_approve_ask: bool = True,
    ) -> ToolResult:
        """Execute tool with full guard evaluation."""
        # Lookup
        tool = self.get(tool_name)
        if tool is None:
            error_msg = f"Tool not found: {tool_name}"
            logger.error(error_msg)
            return ToolResult.fail(error_msg)
        
        # Delegate to executor
        return self._executor.execute(
            tool=tool,
            tool_name=tool_name,
            params=params,
            context=context,
            auto_approve_ask=auto_approve_ask,
        )
    
    def get_execution_history(
        self,
        tool_name: Optional[str] = None,
        series_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get execution history with optional filters."""
        return self._metrics.get_history(tool_name, series_id, limit)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get registry metrics."""
        metrics = self._metrics.get_metrics()
        metrics["registered_tools"] = len(self._tools)
        return metrics


# Global registry instance for convenience
_global_registry: Optional[ToolRegistry] = None


def get_global_registry() -> ToolRegistry:
    """Get or create global tool registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def reset_global_registry() -> None:
    """Reset global registry (mainly for testing)."""
    global _global_registry
    _global_registry = None
