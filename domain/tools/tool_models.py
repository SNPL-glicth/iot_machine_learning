"""Tool data models: context and result.

ToolContext and ToolResult for tool execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ToolContext:
    """Context provided to tools during guard and execution.
    
    Contains all relevant information about the current situation
    for tools to make safety and execution decisions.
    """
    series_id: str
    device_id: Optional[int]
    current_value: float
    severity: str  # CRITICAL, WARNING, NORMAL, etc.
    confidence: float
    trend: str = "stable"
    
    # Historical context
    historical_actions: List[Dict[str, Any]] = field(default_factory=list)
    last_action_time: Optional[datetime] = None
    
    # User/system context
    user_permissions: List[str] = field(default_factory=list)
    maintenance_window: bool = False
    
    def time_since_last_action_seconds(self) -> float:
        """Seconds since last action on this device."""
        if self.last_action_time is None:
            return float('inf')
        return (datetime.utcnow() - self.last_action_time).total_seconds()


@dataclass
class ToolResult:
    """Result of tool execution."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def ok(cls, output: Any = None, metadata: Optional[Dict] = None):
        return cls(
            success=True,
            output=output,
            metadata=metadata or {}
        )
    
    @classmethod
    def fail(cls, error: str, metadata: Optional[Dict] = None):
        return cls(
            success=False,
            error=error,
            metadata=metadata or {}
        )
