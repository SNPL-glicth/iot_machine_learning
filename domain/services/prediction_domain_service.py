"""Servicio de dominio para predicciones.

Orquesta la lógica de predicción usando ports.  No conoce
implementaciones concretas — solo trabaja con interfaces.

Responsabilidades:
- Seleccionar motor de predicción adecuado.
- Aplicar filtro de señal pre-predicción (si está configurado).
- Generar Prediction del dominio.
- Delegar auditoría al AuditPort.
"""

from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from ..entities.prediction import Prediction
from ..entities.sensor_reading import SensorWindow
from ..ports.audit_port import AuditPort
from ..ports.prediction_port import PredictionPort

logger = logging.getLogger(__name__)


class PredictionDomainService:
    """Servicio de dominio que orquesta predicciones.

    Recibe ports por constructor (inyección de dependencias).
    No conoce Taylor, Baseline, SQL Server ni ninguna implementación.

    Attributes:
        _engines: Lista de motores disponibles (ordenados por prioridad).
        _audit: Port de auditoría (puede ser None si no se requiere).
    """

    def __init__(
        self,
        engines: List[PredictionPort],
        audit: Optional[AuditPort] = None,
    ) -> None:
        """Inicializa el servicio con motores y auditoría.

        Args:
            engines: Motores de predicción ordenados por prioridad
                (el primero que pueda manejar los datos se usa).
            audit: Port de auditoría ISO 27001 (opcional).

        Raises:
            ValueError: Si no se proveen motores.
        """
        if not engines:
            raise ValueError("Se requiere al menos un motor de predicción")

        self._engines = engines
        self._audit = audit

    def predict(self, window: SensorWindow) -> Prediction:
        """Genera predicción para una ventana de sensor.

        Selecciona el primer motor que pueda manejar la cantidad de
        datos disponibles.  Si ninguno puede, usa el último como fallback.

        Args:
            window: Ventana temporal del sensor.

        Returns:
            ``Prediction`` del dominio.

        Raises:
            ValueError: Si la ventana está vacía.
        """
        if window.is_empty:
            raise ValueError(
                f"Ventana vacía para sensor {window.sensor_id}"
            )

        trace_id = str(uuid.uuid4())[:12]

        # Seleccionar motor
        engine = self._select_engine(window.size)

        logger.info(
            "prediction_start",
            extra={
                "sensor_id": window.sensor_id,
                "engine": engine.name,
                "n_points": window.size,
                "trace_id": trace_id,
            },
        )

        # Generar predicción
        try:
            prediction = engine.predict(window)

            # Enriquecer con trace_id
            prediction = Prediction(
                sensor_id=prediction.sensor_id,
                predicted_value=prediction.predicted_value,
                confidence_score=prediction.confidence_score,
                trend=prediction.trend,
                engine_name=prediction.engine_name,
                horizon_steps=prediction.horizon_steps,
                confidence_interval=prediction.confidence_interval,
                feature_contributions=prediction.feature_contributions,
                metadata=prediction.metadata,
                audit_trace_id=trace_id,
            )

        except Exception as exc:
            logger.error(
                "prediction_engine_failed",
                extra={
                    "sensor_id": window.sensor_id,
                    "engine": engine.name,
                    "error": str(exc),
                    "trace_id": trace_id,
                },
            )
            # Fallback al último motor (debe ser baseline)
            fallback = self._engines[-1]
            prediction = fallback.predict(window)
            prediction = Prediction(
                sensor_id=prediction.sensor_id,
                predicted_value=prediction.predicted_value,
                confidence_score=prediction.confidence_score,
                trend=prediction.trend,
                engine_name=f"{prediction.engine_name}_fallback",
                horizon_steps=prediction.horizon_steps,
                metadata={**prediction.metadata, "fallback_reason": str(exc)},
                audit_trace_id=trace_id,
            )

        # Auditoría
        if self._audit is not None:
            try:
                self._audit.log_prediction(
                    sensor_id=prediction.sensor_id,
                    predicted_value=prediction.predicted_value,
                    confidence=prediction.confidence_score,
                    engine_name=prediction.engine_name,
                    trace_id=trace_id,
                )
            except Exception:
                logger.warning("audit_log_failed", extra={"trace_id": trace_id})

        logger.info(
            "prediction_complete",
            extra={
                "sensor_id": prediction.sensor_id,
                "predicted_value": prediction.predicted_value,
                "confidence": prediction.confidence_score,
                "engine": prediction.engine_name,
                "trace_id": trace_id,
            },
        )

        return prediction

    def _select_engine(self, n_points: int) -> PredictionPort:
        """Selecciona el primer motor que pueda manejar n_points.

        Args:
            n_points: Puntos disponibles.

        Returns:
            Motor seleccionado (fallback al último si ninguno puede).
        """
        for engine in self._engines:
            if engine.can_handle(n_points):
                return engine

        # Fallback al último (debe ser baseline, que maneja >= 1 punto)
        return self._engines[-1]

    @property
    def available_engines(self) -> List[str]:
        """Nombres de motores disponibles."""
        return [e.name for e in self._engines]
