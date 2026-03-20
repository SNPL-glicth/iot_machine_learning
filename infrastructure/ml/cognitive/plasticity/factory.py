"""Factory for building the AdvancedPlasticityCoordinator and its components.

Extracted from MetaCognitiveOrchestrator.__init__ to keep the orchestrator
under the 300-line complexity guard.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def build_advanced_plasticity(
    storage_adapter=None,
) -> Tuple[object, object, object, object, object]:
    """Instantiate all advanced plasticity components and the coordinator.

    Returns:
        Tuple of (coordinator, adaptive_lr, asymmetric_penalty,
                  contextual_tracker, health_monitor)
    """
    from .adaptive_learning_rate import AdaptiveLearningRate
    from .contextual_plasticity_tracker import ContextualPlasticityTracker
    from ..monitoring.engine_health_monitor import EngineHealthMonitor
    from ....domain.services.asymmetric_penalty_service import AsymmetricPenaltyService
    from .advanced_plasticity_coordinator import AdvancedPlasticityCoordinator

    adaptive_lr = AdaptiveLearningRate()
    asymmetric_penalty = AsymmetricPenaltyService()
    contextual_tracker = ContextualPlasticityTracker()
    health_monitor = EngineHealthMonitor()

    coordinator = AdvancedPlasticityCoordinator(
        adaptive_lr=adaptive_lr,
        asymmetric_penalty=asymmetric_penalty,
        contextual_tracker=contextual_tracker,
        health_monitor=health_monitor,
        storage_adapter=storage_adapter,
    )

    logger.info("advanced_plasticity_enabled")
    return coordinator, adaptive_lr, asymmetric_penalty, contextual_tracker, health_monitor


def null_advanced_plasticity() -> Tuple[None, None, None, None, None]:
    """Return null placeholders when advanced plasticity is disabled."""
    return None, None, None, None, None
