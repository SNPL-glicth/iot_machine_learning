"""Master Decision Port interface.

Defines the contract for the top-level decision orchestrator that computes
the Master Equation and produces actionable execution plans for live trading.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from infrastructure.ml.engines.rosa_roja.algorithms.domain.execution import ExecutionPlan


class MasterDecisionPort(ABC):
    """Abstract Port representing the unique decision entry point for the trading bot.
    
    Orchestrates the Master Equation:
        Φ_RedRose = I(CVaR_t ≤ L_max) · Λ(t) · Φ_MoE_base(RosaRoja)
    
    Decouples market runners and actuators from specific sub-engines.
    """

    @abstractmethod
    def process_event(
        self, delta_state: np.ndarray, delta_time: float
    ) -> ExecutionPlan:
        """Process an incoming state transition ΔS and time delta Δt.

        Evaluates the underlying trajectory generator (Rosa Roja), stochastic risk
        tube, and chronometric synchrony, returning an orchestrated ExecutionPlan.

        Args:
            delta_state: Vector representing state changes (log-return, imbalance, etc.)
            delta_time: Elapsed time interval between updates in seconds.

        Returns:
            ExecutionPlan with the resulting action (EXECUTE, HOLD, EMERGENCY_FLUSH),
            confidence score, action envelope, and comprehensive ISO 22989 decision trace.
        """
        raise NotImplementedError

    @abstractmethod
    def cancel_active_trajectory(self) -> None:
        """Cancel and reset any active tracking trajectory when execution is aborted."""
        raise NotImplementedError

    @property
    @abstractmethod
    def gamma_exec(self) -> float:
        """Execution threshold threshold for confidence gating."""
        raise NotImplementedError

    @property
    @abstractmethod
    def state_machine(self) -> Any:
        """Access the underlying state machine for health and regime monitoring."""
        raise NotImplementedError
