"""Execution Port for Rosa Roja - Domain-Agnostic Action Dispatch."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from domain.entities.rosa_roja.execution import ExecutionPlan


@runtime_checkable
class ExecutionPort(Protocol):
    """
    Native interface implemented by Execution Handlers to process Rosa Roja decisions.
    
    This eliminates the need for ad-hoc translation bridges. The execution layer
    (market, IoT actuator, simulation, etc.) directly implements this protocol
    and receives ExecutionPlan objects natively.
    """
    
    async def dispatch_execution(self, plan: ExecutionPlan) -> bool:
        """Processes an ExecutionPlan directly into domain-specific actions."""
        ...
    
    async def trigger_emergency_flush(self, reason: str) -> None:
        """Triggers emergency cancellation and safety protocol."""
        ...
