"""Prediction service for ML API.

REFACTORIZADO: Delega a la arquitectura hexagonal enterprise.

Antes (God Object — 5 responsabilidades):
  - Carga de datos (SQL)
  - Cálculo (predict_moving_average)
  - Reglas de decisión (threshold evaluation)
  - Persistencia (SQL INSERT)
  - Orquestación de todo lo anterior

Ahora (Thin Orchestrator — 1 responsabilidad):
  - Cablea adapters y delega a PredictSensorValueUseCase
  - Evalúa thresholds delegando a domain/services + repository

Módulos extraídos:
  - infrastructure/adapters/sqlserver_storage.py (StoragePort)
  - infrastructure/ml/engines/baseline_adapter.py (PredictionPort)
  - domain/services/threshold_evaluator.py (reglas puras)
  - infrastructure/repositories/threshold_repository.py (queries)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy.engine import Connection

from iot_machine_learning.application.use_cases.predict_sensor_value import (
    PredictSensorValueUseCase,
)
from iot_machine_learning.domain.services.prediction_domain_service import (
    PredictionDomainService,
)
from iot_machine_learning.domain.services.threshold_evaluator import (
    build_violation,
    is_threshold_violated,
    is_within_warning_range,
)
from iot_machine_learning.infrastructure.adapters.sqlserver_storage import (
    SqlServerStorageAdapter,
)
from iot_machine_learning.infrastructure.ml.engines.baseline_adapter import (
    BaselinePredictionAdapter,
)
from iot_machine_learning.infrastructure.repositories.threshold_repository import (
    ThresholdRepository,
)

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for generating predictions.

    Thin orchestrator: cablea adapters enterprise y delega.
    Mantiene la misma API pública para compatibilidad con routes.py.
    """

    def __init__(self, conn: Connection):
        self._conn = conn

        # --- Wiring enterprise ---
        self._storage = SqlServerStorageAdapter(conn)
        self._threshold_repo = ThresholdRepository(conn)

        baseline_engine = BaselinePredictionAdapter(window=60)
        prediction_domain_service = PredictionDomainService(
            engines=[baseline_engine],
        )

        self._use_case = PredictSensorValueUseCase(
            prediction_service=prediction_domain_service,
            storage=self._storage,
        )

    def predict(
        self,
        *,
        sensor_id: int,
        horizon_minutes: int = 10,
        window: int = 60,
        dedupe_minutes: int = 10,
    ) -> dict:
        """Generate a prediction for a sensor.

        Args:
            sensor_id: ID of the sensor
            horizon_minutes: Prediction horizon in minutes
            window: Number of recent values to use
            dedupe_minutes: Minutes for event deduplication

        Returns:
            dict with prediction details

        Raises:
            ValueError: If no recent readings available
        """
        # 1. Delegar predicción al use case enterprise
        dto = self._use_case.execute(sensor_id=sensor_id, window_size=window)

        # 2. El use case ya persistió la predicción vía StoragePort.
        #    Obtener IDs para compatibilidad con respuesta legacy.
        device_id = self._storage.get_device_id_for_sensor(sensor_id)
        model_id = self._storage._get_or_create_model_id(
            sensor_id, dto.engine_name
        )
        target_ts = datetime.now(timezone.utc) + timedelta(minutes=horizon_minutes)

        # Obtener prediction_id (última predicción insertada)
        latest = self._storage.get_latest_prediction(sensor_id)
        prediction_id = 0  # fallback
        if latest and abs(latest.predicted_value - dto.predicted_value) < 1e-9:
            # Buscar ID real de la predicción recién insertada
            prediction_id = self._get_latest_prediction_id(sensor_id)

        # 3. Evaluar thresholds (reglas de dominio + repository)
        self._eval_thresholds(
            sensor_id=sensor_id,
            device_id=device_id,
            prediction_id=prediction_id,
            predicted_value=dto.predicted_value,
            dedupe_minutes=dedupe_minutes,
        )

        return {
            "sensor_id": sensor_id,
            "model_id": model_id,
            "prediction_id": prediction_id,
            "predicted_value": dto.predicted_value,
            "confidence": dto.confidence_score,
            "target_timestamp": target_ts,
            "horizon_minutes": horizon_minutes,
            "window": window,
        }

    def _eval_thresholds(
        self,
        *,
        sensor_id: int,
        device_id: int,
        prediction_id: int,
        predicted_value: float,
        dedupe_minutes: int,
    ) -> None:
        """Evalúa thresholds delegando a domain rules + repository."""
        # 1. Verificar rango WARNING (domain rule + repo I/O)
        warning_min, warning_max = self._threshold_repo.load_warning_range(sensor_id)
        if is_within_warning_range(predicted_value, warning_min, warning_max):
            return

        # 2. Cargar threshold activo (repo I/O)
        threshold = self._threshold_repo.load_active_threshold(sensor_id)
        if threshold is None:
            return

        # 3. Evaluar violación (domain rule pura)
        if not is_threshold_violated(predicted_value, threshold):
            return

        # 4. Deduplicar (repo I/O)
        if self._threshold_repo.has_recent_event(
            sensor_id, "PRED_THRESHOLD_BREACH", dedupe_minutes
        ):
            return

        # 5. Construir violación (domain rule pura)
        violation = build_violation(predicted_value, threshold)

        # 6. Persistir evento (repo I/O)
        self._threshold_repo.insert_threshold_event(
            sensor_id=sensor_id,
            device_id=device_id,
            prediction_id=prediction_id,
            violation=violation,
        )

    def _get_latest_prediction_id(self, sensor_id: int) -> int:
        """Obtiene el ID de la última predicción insertada."""
        from sqlalchemy import text

        row = self._conn.execute(
            text(
                """
                SELECT TOP 1 id
                FROM dbo.predictions
                WHERE sensor_id = :sensor_id
                ORDER BY predicted_at DESC
                """
            ),
            {"sensor_id": sensor_id},
        ).fetchone()

        return int(row[0]) if row else 0
