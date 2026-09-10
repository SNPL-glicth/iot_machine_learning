"""ExpertJuryPort protocol for MoE experts."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from ..domain.trajectory import Trajectory


@runtime_checkable
class ExpertJuryPort(Protocol):
    """Protocol implemented by MoE engines (Taylor, Kalman, LightGBM, Statistical)."""
    name: str
    is_critical: bool
    threshold: float
    weight: float

    def evaluate_trajectory(self, trajectory: Trajectory) -> float:
        """Evaluates candidate trajectory and returns Ψ_e(T) in [0.0, 1.0]."""
        ...

    def update_learning(self, actual: float, predicted: float) -> None:
        """Called after actual outcome is known for online learning."""
        ...