"""Execution Port for Rosa Roja - Domain-Agnostic Action Dispatch."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from ..domain.execution import ExecutionPlan


@runtime_checkable
class ExecutionPort(Protocol):
    """
    Native interface implemented by Execution Handlers to process Rosa Roja decisions.
    
    This eliminates the need for ad-hoc translation bridges. The execution layer
    (market, IoT actuator, simulation, etc.) directly implements this protocol
    and receives ExecutionPlan objects natively.
    
    Usage in any execution layer:
    
    class MyExecutionHandler:
        def __init__(self, ...):
            ...
        
        def dispatch_execution(self, plan: ExecutionPlan) -> bool:
            if plan.action == "EXECUTE":
                self._actuate(plan)
            elif plan.action == "EMERGENCY_FLUSH":
                self._emergency_shutdown(plan)
            return True
        
        def trigger_emergency_flush(self, reason: str) -> None:
            self._cancel_all_actions()
            self._safe_shutdown()
    """
    
    def dispatch_execution(self, plan: ExecutionPlan) -> bool:
        """
        Processes an ExecutionPlan directly into domain-specific actions.
        
        Args:
            plan: The orchestrated execution plan from Rosa Roja Engine.
            
        Returns:
            True if execution was dispatched successfully, False otherwise.
        """
        ...
    
    def trigger_emergency_flush(self, reason: str) -> None:
        """
        Triggers emergency cancellation and safety protocol.
        
        Called when:
        - Module 1 detects regime change (Mahalanobis outlier)
        - Module 3 hard-gating vetoes all trajectories
        - External safety limits breached
        
        Args:
            reason: Human-readable reason for emergency action.
        """
        ...