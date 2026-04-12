"""Caso de uso: Evaluar thresholds y crear eventos.

Responsabilidad única: Orquestar evaluación de thresholds delegando a:
- domain/services/threshold_evaluator.py (reglas puras)
- infrastructure/repositories/threshold_repository.py (I/O)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iot_machine_learning.infrastructure.repositories.threshold_repository import (
        ThresholdRepository,
    )

from iot_machine_learning.domain.services.threshold_evaluator import (
    build_violation,
    is_threshold_violated,
    is_within_warning_range,
)

logger = logging.getLogger(__name__)


class EvaluateThresholdsUseCase:
    """Evalúa thresholds y crea eventos si hay violación.
    
    Orquesta:
    1. Verificar rango WARNING (domain + repo)
    2. Cargar threshold activo (repo)
    3. Evaluar violación (domain)
    4. Deduplicar (repo)
    5. Construir violación (domain)
    6. Persistir evento (repo)
    """
    
    def __init__(self, threshold_repo: "ThresholdRepository"):
        """Inicializa con repositorio de thresholds.
        
        Args:
            threshold_repo: Repositorio para I/O de thresholds
        """
        self._threshold_repo = threshold_repo
    
    def execute(
        self,
        *,
        sensor_id: int,
        device_id: int,
        prediction_id: int,
        predicted_value: float,
        dedupe_minutes: int = 10,
    ) -> bool:
        """Evalúa thresholds y crea evento si hay violación.
        
        Args:
            sensor_id: ID del sensor
            device_id: ID del dispositivo
            prediction_id: ID de la predicción
            predicted_value: Valor predicho
            dedupe_minutes: Minutos para deduplicación
        
        Returns:
            True si se creó evento, False si no hubo violación
        """
        # 1. Verificar rango WARNING (domain rule + repo I/O)
        warning_min, warning_max = self._threshold_repo.load_warning_range(sensor_id)
        if is_within_warning_range(predicted_value, warning_min, warning_max):
            logger.debug(
                "threshold_within_warning",
                extra={"sensor_id": sensor_id, "value": predicted_value},
            )
            return False
        
        # 2. Cargar threshold activo (repo I/O)
        threshold = self._threshold_repo.load_active_threshold(sensor_id)
        if threshold is None:
            logger.debug(
                "threshold_not_configured",
                extra={"sensor_id": sensor_id},
            )
            return False
        
        # 3. Evaluar violación (domain rule pura)
        if not is_threshold_violated(predicted_value, threshold):
            logger.debug(
                "threshold_not_violated",
                extra={
                    "sensor_id": sensor_id,
                    "value": predicted_value,
                    "threshold": threshold,
                },
            )
            return False
        
        # 4. Deduplicar (repo I/O)
        if self._threshold_repo.has_recent_event(
            sensor_id, "PRED_THRESHOLD_BREACH", dedupe_minutes
        ):
            logger.debug(
                "threshold_event_deduplicated",
                extra={"sensor_id": sensor_id, "dedupe_minutes": dedupe_minutes},
            )
            return False
        
        # 5. Construir violación (domain rule pura)
        violation = build_violation(predicted_value, threshold)
        
        # 6. Persistir evento (repo I/O)
        self._threshold_repo.insert_threshold_event(
            sensor_id=sensor_id,
            device_id=device_id,
            prediction_id=prediction_id,
            violation=violation,
        )
        
        logger.info(
            "threshold_event_created",
            extra={
                "sensor_id": sensor_id,
                "prediction_id": prediction_id,
                "violation": violation,
            },
        )
        
        return True
