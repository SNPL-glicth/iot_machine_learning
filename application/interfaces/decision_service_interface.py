"""Decision service interface (ARCH-CRIT-1).

Applies DIP: Infrastructure depends on this abstraction, not on domain directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DecisionAction(Enum):
    """Decision actions (application layer)."""
    ALERT = "alert"
    NOTIFY = "notify"
    LOG = "log"
    IGNORE = "ignore"


class DecisionPriority(Enum):
    """Decision priorities (application layer)."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class DecisionRequest:
    """Decision request DTO (application layer).
    
    Attributes:
        series_id: Series identifier.
        severity_score: Base severity [0.0, 1.0].
        is_drift: Whether drift detected.
        is_anomaly: Whether anomaly detected.
        confidence: Prediction confidence [0.0, 1.0].
        metadata: Optional additional context.
    """
    series_id: str
    severity_score: float
    is_drift: bool = False
    is_anomaly: bool = False
    confidence: float = 0.5
    metadata: Optional[dict] = None


@dataclass
class DecisionResponse:
    """Decision response DTO (application layer).
    
    Attributes:
        action: Recommended action.
        priority: Action priority.
        final_score: Final computed score [0.0, 1.0].
        reasoning: Human-readable explanation.
    """
    action: DecisionAction
    priority: DecisionPriority
    final_score: float
    reasoning: str


class IDecisionService(ABC):
    """Decision service interface (ARCH-CRIT-1).
    
    Applies DIP: Infrastructure layer depends on this abstraction.
    Application layer provides concrete implementation.
    """
    
    @abstractmethod
    def decide(self, request: DecisionRequest) -> DecisionResponse:
        """Make a decision based on request.
        
        Args:
            request: Decision request with context.
        
        Returns:
            Decision response with action and priority.
        """
        pass
