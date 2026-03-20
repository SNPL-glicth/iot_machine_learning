"""Adapter: wraps MetaCognitiveOrchestrator as PredictionPort.

Bridges the cognitive orchestrator (which implements PredictionEngine)
into the domain's PredictionPort interface, so it can be used by
PredictionDomainService and use cases without modification.

Mapping:
    PredictionEngine.predict(values, timestamps) → PredictionResult
    PredictionPort.predict(window: SensorWindow)  → Prediction
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

from iot_machine_learning.domain.entities.prediction import Prediction
from iot_machine_learning.domain.entities.sensor_reading import SensorWindow
from iot_machine_learning.domain.ports.prediction_port import PredictionPort
from .orchestration import MetaCognitiveOrchestrator

logger = logging.getLogger(__name__)


class CognitivePredictionAdapter(PredictionPort):
    """Adapter exposing MetaCognitiveOrchestrator as PredictionPort.

    .. deprecated:: 2.0
        Use ``orchestrator.as_port()`` or
        ``PredictionEnginePortBridge(orchestrator)`` instead.

    Attributes:
        _orchestrator: The cognitive orchestrator instance.
    """

    def __init__(self, orchestrator: MetaCognitiveOrchestrator) -> None:
        warnings.warn(
            "CognitivePredictionAdapter is deprecated. "
            "Use orchestrator.as_port() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._orchestrator = orchestrator

    @property
    def name(self) -> str:
        return "cognitive_orchestrator"

    def can_handle(self, n_points: int) -> bool:
        return self._orchestrator.can_handle(n_points)

    def predict(self, window: SensorWindow) -> Prediction:
        """Adapt SensorWindow → values/timestamps and PredictionResult → Prediction."""
        values = window.values
        timestamps = window.timestamps if window.timestamps else None

        result = self._orchestrator.predict(values, timestamps)

        return Prediction(
            series_id=str(window.sensor_id),
            predicted_value=result.predicted_value,
            confidence_score=result.confidence,
            trend=result.trend,
            engine_name="cognitive_orchestrator",
            metadata=result.metadata,
        )

    def record_actual(self, actual_value: float) -> None:
        """Forward actual value to orchestrator for plasticity learning."""
        self._orchestrator.record_actual(actual_value)
