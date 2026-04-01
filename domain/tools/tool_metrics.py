"""Tool metrics and history tracking.

Metrics collection for tool registry.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ToolMetrics:
    """Tracks execution metrics for tools."""
    
    def __init__(self, max_history: int = 1000):
        self._execution_history: List[Dict[str, Any]] = []
        self._max_history = max_history
    
    def record_execution(self, entry: Dict[str, Any]) -> None:
        """Record execution to history."""
        self._execution_history.append(entry)
        
        # Trim history if needed
        if len(self._execution_history) > self._max_history:
            self._execution_history = self._execution_history[-self._max_history:]
    
    def get_history(
        self,
        tool_name: Optional[str] = None,
        series_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get execution history with optional filters."""
        history = self._execution_history
        
        if tool_name:
            history = [h for h in history if h["tool_name"] == tool_name]
        
        if series_id:
            history = [h for h in history if h["series_id"] == series_id]
        
        return history[-limit:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Calculate aggregate metrics."""
        if not self._execution_history:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "avg_execution_time_ms": 0.0,
            }
        
        total = len(self._execution_history)
        successful = sum(1 for h in self._execution_history if h["success"])
        avg_time = sum(h["execution_time_ms"] for h in self._execution_history) / total
        
        return {
            "total_executions": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "avg_execution_time_ms": avg_time,
        }
