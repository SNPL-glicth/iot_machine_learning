"""Persistence mixin for PredictSensorValueUseCase.

Handles saving predictions to storage and building PredictionDTOs.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..dto.prediction_dto import PredictionDTO

logger = logging.getLogger(__name__)


class _PredictionPersistenceMixin:
    """Mixin that provides prediction persistence and DTO building."""

    def _persist(self: Any, prediction, series_id: str) -> None:
        try:
            self._storage.save_prediction(prediction)
        except (ConnectionError, TimeoutError, TypeError, AttributeError) as exc:
            logger.warning(
                "prediction_persistence_failed",
                extra={
                    "series_id": series_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "trace_id": prediction.audit_trace_id,
                },
            )

    def _build_dto(
        self: Any,
        prediction,
        memory_context: Optional[dict] = None,
    ) -> PredictionDTO:
        return PredictionDTO(
            series_id=prediction.series_id,
            predicted_value=prediction.predicted_value,
            confidence_score=prediction.confidence_score,
            confidence_level=prediction.confidence_level.value,
            trend=prediction.trend,
            engine_name=prediction.engine_name,
            confidence_interval=prediction.confidence_interval,
            feature_contributions=prediction.feature_contributions,
            audit_trace_id=prediction.audit_trace_id,
            explanation_summary=None,
        )
