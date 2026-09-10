"""ExecutionPlan and ActionEnvelope domain models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any
from .trajectory import Trajectory


@dataclass(frozen=True, slots=True)
class ActionEnvelope:
    """Generic action parameters derived from confidence bands.
    
    Domain-agnostic envelope carrying execution directives.
    Concrete interpretation (position sizing, risk limits, actuator commands)
    is delegated to the outbound adapter layer.
    """
    magnitude: float          # 0.0-1.0 normalized intensity of action
    bounds: dict              # Arbitrary key-value bounds (stop, target, limits, etc.)
    max_steps: int            # Maximum steps/duration for this action
    metadata: dict            # Extension point for domain-specific data
    decision_trace: Optional[dict] = None  # ISO 22989 traceability

    def with_decision_trace(self, trace: dict) -> "ActionEnvelope":
        """Return new envelope with decision trace attached."""
        return ActionEnvelope(
            magnitude=self.magnitude,
            bounds=self.bounds,
            max_steps=self.max_steps,
            metadata=self.metadata,
            decision_trace=trace,
        )


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Final output produced by Rosa Roja for the execution layer."""
    action: str                        # "EXECUTE", "HOLD", "EMERGENCY_FLUSH"
    chosen_trajectory: Optional[Trajectory]
    global_confidence: float           # Phi_MoE score in [0.0, 1.0]
    envelope: Optional[ActionEnvelope] # Action parameters (None for HOLD/FLUSH)
    invalidation_step: Optional[int]   # Index where trajectory is breached
    regime_alert: bool                 # Triggered if Module 1 Mahalanobis outlier detected
    veto_details: dict

    @classmethod
    def HOLD(cls, reason: str, alert: bool = False, details: dict = None) -> "ExecutionPlan":
        return cls(
            action="HOLD",
            chosen_trajectory=None,
            global_confidence=0.0,
            envelope=None,
            invalidation_step=None,
            regime_alert=alert,
            veto_details=details or {"reason": reason}
        )

    @classmethod
    def EXECUTE(cls, trajectory: Trajectory, confidence: float,
                envelope: ActionEnvelope, invalidation_step: Optional[int] = None) -> "ExecutionPlan":
        return cls(
            action="EXECUTE",
            chosen_trajectory=trajectory,
            global_confidence=confidence,
            envelope=envelope,
            invalidation_step=invalidation_step,
            regime_alert=False,
            veto_details={}
        )

    @classmethod
    def EMERGENCY_FLUSH(cls, reason: str) -> "ExecutionPlan":
        return cls(
            action="EMERGENCY_FLUSH",
            chosen_trajectory=None,
            global_confidence=0.0,
            envelope=None,
            invalidation_step=None,
            regime_alert=True,
            veto_details={"reason": reason}
        )