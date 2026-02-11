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
from ...domain.ports.storage_port import StoragePort
from ...domain.services.prediction_domain_service import PredictionDomainService
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
    """

    def __init__(
        self,
        prediction_service: PredictionDomainService,
        storage: StoragePort,
        audit: Optional[AuditPort] = None,
        window_size: int = 500,
    ) -> None:
        self._prediction_service = prediction_service
        self._storage = storage
        self._audit = audit
        self._window_size = window_size

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

        # 4. Construir DTO
        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        dto = PredictionDTO(
            sensor_id=prediction.sensor_id,
            predicted_value=prediction.predicted_value,
            confidence_score=prediction.confidence_score,
            confidence_level=prediction.confidence_level.value,
            trend=prediction.trend,
            engine_name=prediction.engine_name,
            confidence_interval=prediction.confidence_interval,
            feature_contributions=prediction.feature_contributions,
            audit_trace_id=prediction.audit_trace_id,
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
