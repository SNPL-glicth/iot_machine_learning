"""MLflow tracking mixin for PredictSensorValueUseCase.

Fail-safe metric logging to the experiment tracker.
"""

from __future__ import annotations

import logging
from typing import Any

from ..dto.prediction_dto import PredictionDTO

logger = logging.getLogger(__name__)


class _PredictionTrackingMixin:
    """Mixin that provides MLflow-style prediction tracking."""

    def _track_prediction(
        self: Any,
        series_id: str,
        dto: PredictionDTO,
        window_size: int,
        elapsed_ms: float,
    ) -> None:
        """Track prediction metrics to MLflow (fail-safe)."""
        try:
            self._prediction_count += 1

            metrics = {
                "confidence_score": dto.confidence_score,
                "elapsed_ms": elapsed_ms,
            }
            if dto.confidence_interval:
                lower, upper = dto.confidence_interval
                metrics["confidence_interval_width"] = upper - lower

            self._experiment_tracker.log_metrics(
                metrics, step=self._prediction_count
            )
            self._experiment_tracker.log_params({
                "engine_name": dto.engine_name,
                "series_id": dto.series_id,
                "window_size": window_size,
            })
            self._experiment_tracker.set_tags({
                "pipeline_version": "0.2.1-GOLD",
                "series_id": series_id,
            })

        except (ConnectionError, TimeoutError, TypeError, AttributeError) as exc:
            logger.debug(
                "prediction_tracking_failed",
                extra={
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            from ...ml_service.metrics.observability import get_observability
            get_observability().silent_failures.record(
                "prediction_tracking", str(exc), {"series_id": series_id}
            )
