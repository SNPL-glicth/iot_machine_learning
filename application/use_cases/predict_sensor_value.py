"""Caso de uso: Predecir valor de sensor.

Orquesta: cargar datos → predecir → persistir → retornar DTO.
ISO 27001: toda operación genera trace_id auditable.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from ...domain.entities.sensor_reading import SensorReading, SensorWindow
from ...domain.ports.audit_port import AuditPort
from ...domain.ports.cognitive_memory_port import CognitiveMemoryPort
from ...domain.ports.experiment_tracker_port import (
    ExperimentTrackerPort,
    NullExperimentTracker,
)
from ...domain.ports.storage_port import StoragePort
from ...domain.services.memory_recall_enricher import (
    MemoryRecallContext,
    MemoryRecallEnricher,
)
from ...domain.services.prediction_domain_service import PredictionDomainService
from ...domain.validators.data_sanitizer import DataSanitizer
from ...ml_service.config.feature_flags import FeatureFlags
from ..dto.prediction_dto import PredictionDTO

logger = logging.getLogger(__name__)


class PredictSensorValueUseCase:
    """Predicción de sensor. DI puro, sin conocer implementaciones."""

    def __init__(
        self,
        prediction_service: PredictionDomainService,
        storage: StoragePort,
        audit: Optional[AuditPort] = None,
        window_size: int = 500,
        cognitive: Optional[CognitiveMemoryPort] = None,
        flags: Optional[FeatureFlags] = None,
        experiment_tracker: Optional[ExperimentTrackerPort] = None,
    ) -> None:
        self._prediction_service = prediction_service
        self._storage = storage
        self._audit = audit
        self._window_size = window_size
        self._flags = flags
        self._experiment_tracker = experiment_tracker or NullExperimentTracker()
        self._prediction_count = 0
        self._sanitizer = DataSanitizer()

        self._recall_enricher: Optional[MemoryRecallEnricher] = None
        if cognitive is not None:
            self._recall_enricher = MemoryRecallEnricher(cognitive)

    def execute(
        self,
        sensor_id: int,
        window_size: Optional[int] = None,
    ) -> PredictionDTO:
        """Ejecuta predicción cargando datos de storage."""
        t_start = time.monotonic()
        effective_window = window_size or self._window_size

        # 1. Cargar datos
        window = self._storage.load_sensor_window(
            sensor_id=sensor_id,
            limit=effective_window,
        )

        if window.is_empty:
            raise ValueError(
                f"Sensor {sensor_id} no tiene lecturas disponibles"
            )

        # 1.5. Sanitizar datos (boundary de robustez)
        # Intercepta los 9 escenarios de ROBUSTNESS_AUDIT.md
        try:
            sanitized = self._sanitizer.sanitize(
                values=window.values,
                timestamps=window.timestamps,
            )
        except ValueError as e:
            # Re-raise con contexto para que routes.py devuelva 422
            raise ValueError(
                f"Data validation failed for sensor {sensor_id}: {e}"
            ) from e

        # Log warnings no fatales
        if sanitized.warnings:
            logger.warning(
                "input_sanitization_warnings",
                extra={
                    "sensor_id": sensor_id,
                    "warnings": sanitized.warnings,
                },
            )

        # Reconstruir ventana con datos sanitizados
        sanitized_window = SensorWindow(
            series_id=window.series_id,
            readings=[
                SensorReading(series_id=window.series_id, value=v, timestamp=ts)
                for v, ts in zip(sanitized.values, sanitized.timestamps)
            ],
        )

        logger.info(
            "use_case_predict_start",
            extra={
                "sensor_id": sensor_id,
                "window_size": sanitized_window.size,
                "sanitization_warnings": len(sanitized.warnings),
            },
        )

        # 2. Generar predicción
        prediction = self._prediction_service.predict(sanitized_window)

        # 3. Persistir + build DTO
        self._persist(prediction, sensor_id)

        # 3.5. Memory recall enrichment (optional, fail-safe)
        memory_context = self._try_recall(prediction, sensor_id)

        elapsed_ms = (time.monotonic() - t_start) * 1000.0
        dto = self._build_dto(prediction, memory_context)

        # 4. MLflow tracking (fail-safe)
        self._track_prediction(sensor_id, dto, window.size, elapsed_ms)

        logger.info(
            "use_case_predict_complete",
            extra={
                "sensor_id": sensor_id,
                "predicted_value": prediction.predicted_value,
                "engine": prediction.engine_name,
                "elapsed_ms": round(elapsed_ms, 2),
                "trace_id": prediction.audit_trace_id,
            },
        )
        return dto

    def execute_with_window(self, sensor_window: SensorWindow) -> PredictionDTO:
        """Execute prediction with a pre-loaded SensorWindow (no SQL reload)."""
        t_start = time.monotonic()
        if sensor_window.is_empty:
            raise ValueError(
                f"Sensor {sensor_window.sensor_id} no tiene lecturas disponibles"
            )

        # Sanitizar datos (boundary de robustez)
        try:
            sanitized = self._sanitizer.sanitize(
                values=sensor_window.values,
                timestamps=sensor_window.timestamps,
            )
        except ValueError as e:
            raise ValueError(
                f"Data validation failed for sensor {sensor_window.sensor_id}: {e}"
            ) from e

        if sanitized.warnings:
            logger.warning(
                "input_sanitization_warnings",
                extra={
                    "sensor_id": sensor_window.sensor_id,
                    "warnings": sanitized.warnings,
                },
            )

        sanitized_window = SensorWindow(
            sensor_id=sensor_window.sensor_id,
            values=sanitized.values,
            timestamps=sanitized.timestamps,
        )

        logger.info(
            "use_case_predict_preloaded_start",
            extra={
                "sensor_id": sensor_window.sensor_id,
                "window_size": sanitized_window.size,
                "sanitization_warnings": len(sanitized.warnings),
            },
        )
        prediction = self._prediction_service.predict(sanitized_window)
        self._persist(prediction, sensor_window.sensor_id)
        elapsed_ms = (time.monotonic() - t_start) * 1000.0
        dto = self._build_dto(prediction)
        logger.info(
            "use_case_predict_preloaded_complete",
            extra={
                "sensor_id": sensor_window.sensor_id,
                "predicted_value": prediction.predicted_value,
                "engine": prediction.engine_name,
                "elapsed_ms": round(elapsed_ms, 2),
                "trace_id": prediction.audit_trace_id,
            },
        )
        return dto

    # -- private helpers --------------------------------------------------

    def _persist(self, prediction, sensor_id) -> None:
        try:
            self._storage.save_prediction(prediction)
        except Exception as exc:
            logger.error(
                "prediction_persistence_failed",
                extra={"sensor_id": sensor_id, "error": str(exc),
                       "trace_id": prediction.audit_trace_id},
            )

    def _build_dto(self, prediction, memory_context=None) -> PredictionDTO:
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
            memory_context=memory_context,
        )

    def _try_recall(self, prediction, sensor_id):
        if not self._should_recall():
            return None
        try:
            recall_ctx = self._recall_enricher.enrich(prediction)
            if recall_ctx.has_context:
                return recall_ctx.to_dict()
        except Exception as exc:
            logger.warning(
                "memory_recall_failed",
                extra={"sensor_id": sensor_id, "error": str(exc)},
            )
            # OBSERVABILITY: Track silent failure in memory recall
            from ...ml_service.metrics.observability import get_observability
            get_observability().silent_failures.record(
                "memory_recall", str(exc), {"sensor_id": sensor_id}
            )
        return None

    def _should_recall(self) -> bool:
        """Check if memory recall is enabled and available."""
        if self._recall_enricher is None:
            return False
        if self._flags is None:
            return False
        return self._flags.ML_ENABLE_MEMORY_RECALL

    def _track_prediction(
        self,
        sensor_id: int,
        dto: PredictionDTO,
        window_size: int,
        elapsed_ms: float,
    ) -> None:
        """Track prediction metrics to MLflow (fail-safe)."""
        try:
            self._prediction_count += 1

            # Log metrics
            metrics = {
                "confidence_score": dto.confidence_score,
                "elapsed_ms": elapsed_ms,
            }
            if dto.confidence_interval:
                lower, upper = dto.confidence_interval
                metrics["confidence_interval_width"] = upper - lower

            self._experiment_tracker.log_metrics(metrics, step=self._prediction_count)

            # Log params
            self._experiment_tracker.log_params({
                "engine_name": dto.engine_name,
                "series_id": dto.series_id,
                "window_size": window_size,
            })

            # Log tags
            self._experiment_tracker.set_tags({
                "pipeline_version": "0.2.1-GOLD",
                "sensor_id": str(sensor_id),
            })

        except Exception as exc:
            # Fail-safe: never break pipeline for tracking errors
            logger.debug("prediction_tracking_failed", extra={"error": str(exc)})
            # OBSERVABILITY: Track silent failure in tracking
            from ...ml_service.metrics.observability import get_observability
            get_observability().silent_failures.record(
                "prediction_tracking", str(exc), {"sensor_id": sensor_id}
            )
