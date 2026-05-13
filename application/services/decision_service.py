"""Decision service implementation (ARCH-CRIT-1).

Application layer service that bridges domain and infrastructure.

Applies DIP: This is the concrete implementation that infrastructure uses.
"""

from __future__ import annotations

import logging
from typing import Optional

from iot_machine_learning.domain.entities.decision import Decision, DecisionContext
from iot_machine_learning.domain.ports.decision_port import DecisionEnginePort
from iot_machine_learning.application.interfaces.decision_service_interface import (
    IDecisionService,
    DecisionRequest,
    DecisionResponse,
    DecisionAction,
    DecisionPriority,
)

logger = logging.getLogger(__name__)


class DecisionService(IDecisionService):
    """Decision service implementation (ARCH-CRIT-1).
    
    Bridges application DTOs and domain entities.
    
    Attributes:
        _engine: Domain decision engine port.
    
    Applies DIP: Application → Domain (via port).
    Applies SRP: Only translates between layers, no business logic.
    """
    
    def __init__(self, engine: DecisionEnginePort) -> None:
        """Initialize service.
        
        Args:
            engine: Domain decision engine implementation.
        """
        self._engine = engine
    
    def decide(self, request: DecisionRequest) -> DecisionResponse:
        """Make a decision based on request.
        
        Args:
            request: Application layer request DTO.
        
        Returns:
            Application layer response DTO.
        
        Applies SRP: Translates DTOs to domain entities and back.
        """
        # Translate application DTO to domain entity
        context = DecisionContext(
            series_id=request.series_id,
            severity_score=request.severity_score,
            is_drift=request.is_drift,
            is_anomaly=request.is_anomaly,
            confidence=request.confidence,
            metadata=request.metadata or {},
        )
        
        # Call domain engine
        domain_decision: Decision = self._engine.decide(context)
        
        # Translate domain entity to application DTO
        response = DecisionResponse(
            action=self._map_action(domain_decision.action),
            priority=self._map_priority(domain_decision.priority),
            final_score=domain_decision.score,
            reasoning=domain_decision.reasoning,
        )
        
        return response
    
    def _map_action(self, domain_action: str) -> DecisionAction:
        """Map domain action to application action.
        
        Args:
            domain_action: Domain action string.
        
        Returns:
            Application DecisionAction enum.
        """
        mapping = {
            "alert": DecisionAction.ALERT,
            "notify": DecisionAction.NOTIFY,
            "log": DecisionAction.LOG,
            "ignore": DecisionAction.IGNORE,
        }
        return mapping.get(domain_action.lower(), DecisionAction.LOG)
    
    def _map_priority(self, domain_priority: str) -> DecisionPriority:
        """Map domain priority to application priority.
        
        Args:
            domain_priority: Domain priority string.
        
        Returns:
            Application DecisionPriority enum.
        """
        mapping = {
            "critical": DecisionPriority.CRITICAL,
            "high": DecisionPriority.HIGH,
            "medium": DecisionPriority.MEDIUM,
            "low": DecisionPriority.LOW,
        }
        return mapping.get(domain_priority.lower(), DecisionPriority.LOW)
