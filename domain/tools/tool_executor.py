"""Tool executor with safety guards.

Execution logic for tools with guard evaluation.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict

from .tool_base import Tool
from .tool_guard import SafetyLevel
from .tool_models import ToolContext, ToolResult

if TYPE_CHECKING:
    from .tool_metrics import ToolMetrics

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Execute tools with full guard evaluation."""
    
    def __init__(self, metrics: "ToolMetrics"):
        self._metrics = metrics
    
    def execute(
        self,
        tool: Tool,
        tool_name: str,
        params: Dict[str, Any],
        context: ToolContext,
        auto_approve_ask: bool = True,
    ) -> ToolResult:
        """Execute single tool with guard chain.
        
        Returns:
            ToolResult from execution or guard denial
        """
        start_time = time.time()
        
        # Step 1: Validate parameters
        is_valid, error = tool.validate_params(params)
        if not is_valid:
            logger.warning(f"Tool '{tool_name}' param validation failed: {error}")
            return ToolResult.fail(f"Parameter validation: {error}")
        
        # Step 2: Evaluate guard
        guard_result = tool.can_execute(context)
        
        # Step 3: Handle DENY
        if not guard_result.allowed or guard_result.level == SafetyLevel.DENY:
            logger.warning(
                f"Tool '{tool_name}' DENIED: {guard_result.reason}",
                extra={
                    "series_id": context.series_id,
                    "guard_reason": guard_result.reason,
                }
            )
            return ToolResult.fail(
                f"Guard denied: {guard_result.reason}",
                metadata={"guard_result": "deny", "reason": guard_result.reason}
            )
        
        # Step 4: Handle ASK
        if guard_result.level == SafetyLevel.ASK:
            logger.warning(
                f"Tool '{tool_name}' REQUIRES APPROVAL: {guard_result.reason}",
                extra={
                    "series_id": context.series_id,
                    "auto_approved": auto_approve_ask,
                }
            )
            if not auto_approve_ask:
                return ToolResult.fail(
                    "Awaiting human approval",
                    metadata={
                        "guard_result": "ask",
                        "reason": guard_result.reason,
                        "pending_approval": True,
                    }
                )
        
        # Step 5: Execute
        try:
            exec_start = time.time()
            result = tool.execute(params, context)
            exec_time_ms = (time.time() - exec_start) * 1000
            
            # Update result with execution time if not set
            if result.execution_time_ms == 0.0:
                result = ToolResult(
                    success=result.success,
                    output=result.output,
                    error=result.error,
                    execution_time_ms=exec_time_ms,
                    metadata=result.metadata,
                )
            
            total_time_ms = (time.time() - start_time) * 1000
            self._log_execution(
                tool_name=tool_name,
                context=context,
                guard_result=guard_result,
                result=result,
                total_time_ms=total_time_ms,
            )
            
            return result
            
        except Exception as e:
            logger.exception(f"Tool '{tool_name}' execution crashed")
            return ToolResult.fail(f"Execution exception: {str(e)}")
    
    def _log_execution(
        self,
        tool_name: str,
        context: ToolContext,
        guard_result,
        result: ToolResult,
        total_time_ms: float,
    ) -> None:
        """Log execution to metrics."""
        entry = {
            "timestamp": time.time(),
            "tool_name": tool_name,
            "series_id": context.series_id,
            "guard_level": guard_result.level.name,
            "guard_reason": guard_result.reason,
            "success": result.success,
            "execution_time_ms": result.execution_time_ms,
            "total_time_ms": total_time_ms,
        }
        
        self._metrics.record_execution(entry)
