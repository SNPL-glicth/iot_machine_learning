"""Factory for building the AdvancedBayesianCoordinator and its components.

Extracted from MetaCognitiveOrchestrator.__init__ to keep the orchestrator
under the 300-line complexity guard.

Renamed from 'plasticity' to 'bayesian' for honest naming.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def build_advanced_bayesian(
    storage_adapter=None,
) -> Tuple[object, object, object, object, object]:
    """Instantiate all advanced Bayesian weight tracker components and the coordinator.

    Returns:
        Tuple of (coordinator, adaptive_lr, asymmetric_penalty,
                  contextual_tracker, health_monitor)
    """
    from .adaptive_learning_rate import AdaptiveLearningRate
    from .contextual_weight_tracker import ContextualWeightTracker
    from ..monitoring.engine_health_monitor import EngineHealthMonitor
    from iot_machine_learning.domain.services.asymmetric_penalty_service import AsymmetricPenaltyService
    from .advanced_bayesian_coordinator import AdvancedBayesianCoordinator

    adaptive_lr = AdaptiveLearningRate()
    asymmetric_penalty = AsymmetricPenaltyService()
    contextual_tracker = ContextualWeightTracker()
    health_monitor = EngineHealthMonitor()

    coordinator = AdvancedBayesianCoordinator(
        adaptive_lr=adaptive_lr,
        asymmetric_penalty=asymmetric_penalty,
        contextual_tracker=contextual_tracker,
        health_monitor=health_monitor,
        storage_adapter=storage_adapter,
    )

    logger.info("advanced_bayesian_enabled")
    return coordinator, adaptive_lr, asymmetric_penalty, contextual_tracker, health_monitor


def null_advanced_bayesian() -> Tuple[None, None, None, None, None]:
    """Return null placeholders when advanced Bayesian weight tracker is disabled."""
    return None, None, None, None, None
