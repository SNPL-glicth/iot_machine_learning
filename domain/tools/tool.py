"""Tool abstraction layer facade - re-exports for backward compatibility.

This module re-exports all tool components from their split locations.
For new code, import directly from the specific modules.
"""

from __future__ import annotations

# Re-export from split modules for backward compatibility
from .tool_base import Tool
from .tool_guard import SafetyLevel, GuardResult
from .tool_models import ToolContext, ToolResult

__all__ = [
    "Tool",
    "SafetyLevel",
    "GuardResult", 
    "ToolContext",
    "ToolResult",
]
