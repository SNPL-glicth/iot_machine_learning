"""Decision Service Port (ARCH-CRIT-1).

Interface for decision-making services following DIP.
Infrastructure depends on this abstraction, not on concrete implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iot_machine_learning.domain.entities.decision import Decision, DecisionContext


class IDecisionService(ABC):
    """Abstract interface for decision services (ARCH-CRIT-1).
    
    Applies DIP: Infrastructure layer depends on this abstraction.
    Application layer provides concrete implementation.
    
    Dependency flow:
        infrastructure → IDecisionService ← DecisionService → domain
    """
    
    @abstractmethod
    def decide(self, context: DecisionContext) -> Decision:
        """Make a decision based on context.
        
        Args:
            context: Decision context with severity, regime, drift, etc.
        
        Returns:
            Decision with action, priority, and reasoning.
        
        Raises:
            ValueError: If context is invalid or incomplete.
        """
        pass
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Get the decision strategy name."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Get the service version."""
        pass
