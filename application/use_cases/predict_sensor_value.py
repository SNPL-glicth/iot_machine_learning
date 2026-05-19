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

from ._prediction_execution_mixin import _PredictionExecutionMixin
from ._prediction_persistence_mixin import _PredictionPersistenceMixin
from ._prediction_recall_mixin import _PredictionRecallMixin
from ._prediction_tracking_mixin import _PredictionTrackingMixin

logger = logging.getLogger(__name__)


class PredictSensorValueUseCase(
    _PredictionExecutionMixin,
    _PredictionPersistenceMixin,
    _PredictionRecallMixin,
    _PredictionTrackingMixin,
):
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
        series_id: str,
        window_size: Optional[int] = None,
    ) -> PredictionDTO:
        """Ejecuta predicción cargando datos de storage."""
        t_start = time.monotonic()
        effective_window = window_size or self._window_size

        # 1. Cargar datos
        window = self._storage.load_series_window(
            series_id=series_id,
            limit=effective_window,
        )

        if window.is_empty:
            raise ValueError(
                f"Series {series_id} no tiene lecturas disponibles"
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
                f"Data validation failed for series {series_id}: {e}"
            ) from e

        # Log warnings no fatales
        if sanitized.warnings:
            logger.warning(
                "input_sanitization_warnings",
                extra={
                    "series_id": series_id,
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
                "series_id": series_id,
                "window_size": sanitized_window.size,
                "sanitization_warnings": len(sanitized.warnings),
            },
        )

        # 2. Generar predicción
        prediction = self._prediction_service.predict(sanitized_window)

        # 3. Persistir + build DTO
        self._persist(prediction, series_id)

        # 3.5. Memory recall enrichment (optional, fail-safe)
        memory_context = self._try_recall(prediction, series_id)

        elapsed_ms = (time.monotonic() - t_start) * 1000.0
        dto = self._build_dto(prediction, memory_context)

        # 4. MLflow tracking (fail-safe)
        self._track_prediction(series_id, dto, window.size, elapsed_ms)

        logger.info(
            "use_case_predict_complete",
            extra={
                "series_id": series_id,
                "predicted_value": prediction.predicted_value,
                "engine": prediction.engine_name,
                "elapsed_ms": round(elapsed_ms, 2),
                "trace_id": prediction.audit_trace_id,
            },
        )
        return dto
