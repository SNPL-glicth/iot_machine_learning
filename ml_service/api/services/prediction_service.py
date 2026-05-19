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
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from sqlalchemy.engine import Connection

from iot_machine_learning.application.use_cases.predict_sensor_value import (
    PredictSensorValueUseCase,
)
from iot_machine_learning.application.use_cases.evaluate_thresholds import (
    EvaluateThresholdsUseCase,
)
from iot_machine_learning.application.use_cases.enrich_prediction import (
    EnrichPredictionUseCase,
)
from iot_machine_learning.domain.services.prediction_domain_service import (
    PredictionDomainService,
)
from iot_machine_learning.domain.ports.storage_port import StoragePort
from iot_machine_learning.infrastructure.ml.engines.core.factory import EngineFactory
from iot_machine_learning.infrastructure.repositories.prediction_repository import (
    PredictionRepository,
)
from ml_service.config.feature_flags import get_feature_flags

logger = logging.getLogger(__name__)


class _CognitivePredictionPort:
    """Adapter: MetaCognitiveOrchestrator → PredictionPort for domain service.

    Wraps the orchestrator so it can be injected into PredictionDomainService
    as a first-class engine.  Handles the flags_snapshot requirement that
    the generic PredictionEnginePortBridge does not pass.
    """

    def __init__(self, orchestrator, flags):
        self._orchestrator = orchestrator
        self._flags = flags

    @property
    def name(self) -> str:
        return "meta_cognitive_orchestrator"

    def can_handle(self, n_points: int) -> bool:
        return self._orchestrator.can_handle(n_points)

    def predict(self, window):
        from iot_machine_learning.domain.entities.prediction import Prediction

        result = self._orchestrator.predict(
            series_id=str(window.sensor_id),
            values=window.values,
            timestamps=window.timestamps if window.timestamps else None,
            flags_snapshot=self._flags,
        )
        return Prediction(
            series_id=str(window.sensor_id),
            predicted_value=result.predicted_value,
            confidence_score=result.confidence,
            trend=result.trend,
            engine_name="meta_cognitive_orchestrator",
            metadata=result.metadata,
        )

    def supports_confidence_interval(self) -> bool:
        return False


class PredictionService:
    """Service for generating predictions.

    Thin orchestrator: cablea adapters enterprise y delega.
    Mantiene la misma API pública para compatibilidad con routes.py.
    """

    def __init__(
        self,
        conn: Connection,
        storage: Optional[StoragePort] = None,
        threshold_repo: Optional["ThresholdRepositoryPort"] = None,
        cognitive_orchestrator: Optional[Any] = None,
    ):
        self._conn = conn

        # --- Wiring enterprise with dependency injection ---
        # If storage is not provided, create default (backward compatibility)
        if storage is None:
            from iot_machine_learning.infrastructure.persistence.sql.storage import (
                SqlServerStorageAdapter,
            )
            self._storage: StoragePort = SqlServerStorageAdapter(conn)
        else:
            self._storage = storage

        # If threshold_repo is not provided, create default (backward compatibility)
        if threshold_repo is None:
            from iot_machine_learning.infrastructure.repositories.threshold_repository import (
                ThresholdRepository,
            )
            self._threshold_repo = ThresholdRepository(conn)
        else:
            self._threshold_repo = threshold_repo

        # Initialize use cases
        engines = []
        self._cognitive_orchestrator = cognitive_orchestrator
        if cognitive_orchestrator is not None:
            flags = get_feature_flags()
            engines.append(
                _CognitivePredictionPort(cognitive_orchestrator, flags)
            )
            logger.info(
                "cognitive_orchestrator_injected",
                extra={"engine": "meta_cognitive_orchestrator"},
            )

        baseline_engine = EngineFactory.create("baseline_moving_average")
        kalman_engine = EngineFactory.create("kalman")
        engines.extend([
            kalman_engine.as_port(),
            baseline_engine.as_port(),
        ])

        prediction_domain_service = PredictionDomainService(
            engines=engines,
        )

        self._predict_use_case = PredictSensorValueUseCase(
            prediction_service=prediction_domain_service,
            storage=self._storage,
        )
        self._evaluate_thresholds_use_case = EvaluateThresholdsUseCase(
            threshold_repo=self._threshold_repo,
        )
        self._enrich_prediction_use_case = EnrichPredictionUseCase(
            storage=self._storage,
        )
        self._prediction_repo = PredictionRepository(conn)

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
        t_start = time.monotonic()

        # 1. Delegar predicción al use case enterprise
        dto = self._predict_use_case.execute(series_id=str(sensor_id), window_size=window)

        # 2. El use case ya persistió la predicción vía StoragePort.
        #    Obtener IDs para compatibilidad con respuesta legacy.
        device_id = self._storage.get_device_id_for_sensor(sensor_id)
        model_id = self._storage._predictions._get_or_create_model_id(
            sensor_id, dto.engine_name
        )
        target_ts = datetime.now(timezone.utc) + timedelta(minutes=horizon_minutes)

        # Obtener prediction_id (última predicción insertada)
        prediction_id = self._prediction_repo.get_latest_prediction_id(sensor_id)

        # 3. Evaluar thresholds (delegado a use case) solo si prediction_id es válido
        if prediction_id > 0:
            self._evaluate_thresholds_use_case.execute(
                sensor_id=sensor_id,
                device_id=device_id,
                prediction_id=prediction_id,
                predicted_value=dto.predicted_value,
                dedupe_minutes=dedupe_minutes,
            )

        # 4. Enriquecimiento cognitivo (delegado a use case)
        enrichment = self._enrich_prediction_use_case.execute(
            sensor_id=sensor_id,
            window_size=window,
            dto=dto,
        )

        elapsed_ms = round((time.monotonic() - t_start) * 1000.0, 2)

        # 5. Audit log estructurado
        logger.info(
            "prediction_enriched",
            extra={
                "sensor_id": sensor_id,
                "predicted_value": dto.predicted_value,
                "confidence": dto.confidence_score,
                "trend": dto.trend,
                "engine_used": dto.engine_name,
                "regime": enrichment.get("structural_analysis", {}).get("regime"),
                "certainty": enrichment.get("metacognitive", {}).get("certainty"),
                "elapsed_ms": elapsed_ms,
                "trace_id": dto.audit_trace_id,
            },
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
            # --- Enrichment fields ---
            "trend": dto.trend,
            "engine_used": dto.engine_name,
            "confidence_level": dto.confidence_level,
            "structural_analysis": enrichment.get("structural_analysis"),
            "metacognitive": enrichment.get("metacognitive"),
            "audit_trace_id": dto.audit_trace_id,
            "explanation_summary": dto.explanation_summary,
            "processing_time_ms": elapsed_ms,
        }

