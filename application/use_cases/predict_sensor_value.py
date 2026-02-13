"""Caso de uso: Predecir valor de sensor.

Orquesta el flujo completo de predicción:
1. Cargar ventana de datos del sensor (vía StoragePort).
2. Aplicar filtro de señal si está configurado.
3. Generar predicción (vía PredictionDomainService).
4. Persistir resultado (vía StoragePort).
5. Retornar DTO para la capa de presentación.

Cumple ISO 27001: toda operación genera trace_id auditable.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from ...domain.entities.sensor_reading import SensorWindow
from ...domain.ports.audit_port import AuditPort
from ...domain.ports.cognitive_memory_port import CognitiveMemoryPort
from ...domain.ports.storage_port import StoragePort
from ...domain.services.memory_recall_enricher import (
    MemoryRecallContext,
    MemoryRecallEnricher,
)
from ...domain.services.prediction_domain_service import PredictionDomainService
from ...ml_service.config.feature_flags import FeatureFlags
from ..dto.prediction_dto import PredictionDTO

logger = logging.getLogger(__name__)


class PredictSensorValueUseCase:
    """Caso de uso para predicción de valor de sensor.

    Recibe dependencias por constructor (inyección).
    No conoce SQL Server, Redis ni implementaciones concretas.

    Attributes:
        _prediction_service: Servicio de dominio de predicción.
        _storage: Port de almacenamiento.
        _audit: Port de auditoría (opcional).
        _window_size: Tamaño de ventana por defecto.
        _recall_enricher: Memory recall enricher (optional, created if cognitive port provided).
        _flags: Feature flags (optional, controls memory recall).
    """

    def __init__(
        self,
        prediction_service: PredictionDomainService,
        storage: StoragePort,
        audit: Optional[AuditPort] = None,
        window_size: int = 500,
        cognitive: Optional[CognitiveMemoryPort] = None,
        flags: Optional[FeatureFlags] = None,
    ) -> None:
        self._prediction_service = prediction_service
        self._storage = storage
        self._audit = audit
        self._window_size = window_size
        self._flags = flags

        self._recall_enricher: Optional[MemoryRecallEnricher] = None
        if cognitive is not None:
            self._recall_enricher = MemoryRecallEnricher(cognitive)

    def execute(
        self,
        sensor_id: int,
        window_size: Optional[int] = None,
    ) -> PredictionDTO:
        """Ejecuta el caso de uso de predicción.

        Args:
            sensor_id: ID del sensor a predecir.
            window_size: Tamaño de ventana (override del default).

        Returns:
            ``PredictionDTO`` con resultado de predicción.

        Raises:
            ValueError: Si el sensor no tiene datos.
        """
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

        logger.info(
            "use_case_predict_start",
            extra={
                "sensor_id": sensor_id,
                "window_size": window.size,
            },
        )

        # 2. Generar predicción
        prediction = self._prediction_service.predict(window)

        # 3. Persistir
        try:
            self._storage.save_prediction(prediction)
        except Exception as exc:
            logger.error(
                "prediction_persistence_failed",
                extra={
                    "sensor_id": sensor_id,
                    "error": str(exc),
                    "trace_id": prediction.audit_trace_id,
                },
            )
            # No fallar el use case por error de persistencia

        # 3.5. Memory recall enrichment (optional, fail-safe)
        memory_context = None
        if self._should_recall():
            try:
                recall_ctx = self._recall_enricher.enrich(prediction)
                if recall_ctx.has_context:
                    memory_context = recall_ctx.to_dict()
                    logger.debug(
                        "memory_recall_enriched",
                        extra={
                            "series_id": prediction.series_id,
                            "explanations": len(recall_ctx.similar_explanations),
                            "anomalies": len(recall_ctx.similar_anomalies),
                        },
                    )
            except Exception as exc:
                logger.warning(
                    "memory_recall_failed",
                    extra={
                        "sensor_id": sensor_id,
                        "error": str(exc),
                    },
                )

        # 4. Construir DTO
        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        dto = PredictionDTO(
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

    def _should_recall(self) -> bool:
        """Check if memory recall is enabled and available."""
        if self._recall_enricher is None:
            return False
        if self._flags is None:
            return False
        return self._flags.ML_ENABLE_MEMORY_RECALL
