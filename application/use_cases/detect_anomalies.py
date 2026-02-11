"""Caso de uso: Detectar anomalías en sensor.

Orquesta:
1. Cargar ventana de datos (vía StoragePort).
2. Ejecutar detección de anomalías (vía AnomalyDomainService).
3. Persistir evento si es anomalía (vía StoragePort).
4. Retornar DTO para la capa de presentación.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from ...domain.ports.audit_port import AuditPort
from ...domain.ports.storage_port import StoragePort
from ...domain.services.anomaly_domain_service import AnomalyDomainService
from ..dto.prediction_dto import AnomalyDTO

logger = logging.getLogger(__name__)


class DetectAnomaliesUseCase:
    """Caso de uso para detección de anomalías.

    Attributes:
        _anomaly_service: Servicio de dominio de anomalías.
        _storage: Port de almacenamiento.
        _audit: Port de auditoría (opcional).
        _window_size: Tamaño de ventana por defecto.
    """

    def __init__(
        self,
        anomaly_service: AnomalyDomainService,
        storage: StoragePort,
        audit: Optional[AuditPort] = None,
        window_size: int = 500,
    ) -> None:
        self._anomaly_service = anomaly_service
        self._storage = storage
        self._audit = audit
        self._window_size = window_size

    def execute(
        self,
        sensor_id: int,
        window_size: Optional[int] = None,
    ) -> AnomalyDTO:
        """Ejecuta detección de anomalías para un sensor.

        Args:
            sensor_id: ID del sensor.
            window_size: Override del tamaño de ventana.

        Returns:
            ``AnomalyDTO`` con resultado.
        """
        t_start = time.monotonic()
        effective_window = window_size or self._window_size

        # 1. Cargar datos
        window = self._storage.load_sensor_window(
            sensor_id=sensor_id,
            limit=effective_window,
        )

        if window.is_empty:
            return AnomalyDTO(
                series_id=str(sensor_id),
                is_anomaly=False,
                score=0.0,
                severity="none",
                explanation="Sin datos disponibles",
            )

        # 2. Detectar
        result = self._anomaly_service.detect(window)

        # 3. Persistir si es anomalía
        if result.is_anomaly:
            try:
                self._storage.save_anomaly_event(result)
            except Exception as exc:
                logger.error(
                    "anomaly_persistence_failed",
                    extra={
                        "sensor_id": sensor_id,
                        "error": str(exc),
                        "trace_id": result.audit_trace_id,
                    },
                )

        # 4. DTO
        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        logger.info(
            "use_case_detect_anomalies_complete",
            extra={
                "sensor_id": sensor_id,
                "is_anomaly": result.is_anomaly,
                "score": round(result.score, 4),
                "elapsed_ms": round(elapsed_ms, 2),
            },
        )

        return AnomalyDTO(
            series_id=result.series_id,
            is_anomaly=result.is_anomaly,
            score=result.score,
            severity=result.severity.value,
            method_votes=result.method_votes,
            explanation=result.explanation,
            audit_trace_id=result.audit_trace_id,
        )
