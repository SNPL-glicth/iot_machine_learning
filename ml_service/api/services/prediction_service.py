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

from iot_machine_learning.application.explainability.explanation_renderer import (
    ExplanationRenderer,
)
from iot_machine_learning.application.use_cases.predict_sensor_value import (
    PredictSensorValueUseCase,
)
from iot_machine_learning.domain.entities.explainability.explanation import (
    Explanation,
    Outcome,
)
from iot_machine_learning.domain.entities.explainability.signal_snapshot import (
    SignalSnapshot,
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
        self._renderer = ExplanationRenderer()

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
        t_start = time.monotonic()

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

        # 4. Enriquecimiento cognitivo (fail-safe)
        enrichment = self._compute_enrichment(
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
            "processing_time_ms": elapsed_ms,
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

    def _compute_enrichment(
        self,
        *,
        sensor_id: int,
        window_size: int,
        dto: "PredictionDTO",
    ) -> Dict[str, Any]:
        """Computa structural analysis + metacognitive classification.

        Fail-safe: si algo falla, retorna dict vacío sin romper el flujo.
        No modifica domain ni engines — solo lee datos ya computados.
        """
        try:
            # Cargar ventana para structural analysis
            sensor_window = self._storage.load_sensor_window(
                sensor_id=sensor_id,
                limit=window_size,
            )
            if sensor_window.is_empty:
                return {}

            # Compute structural analysis (domain pure function)
            sa = sensor_window.structural_analysis
            structural_dict = {
                "regime": sa.regime.value,
                "slope": round(sa.slope, 6),
                "curvature": round(sa.curvature, 6),
                "noise_ratio": round(sa.noise_ratio, 6),
                "stability": round(sa.stability, 6),
                "trend_strength": round(sa.trend_strength, 6),
                "mean": round(sa.mean, 6),
                "std": round(sa.std, 6),
                "n_points": sa.n_points,
            }

            # Build minimal Explanation for metacognitive classification
            signal = SignalSnapshot(
                n_points=sa.n_points,
                mean=sa.mean,
                std=sa.std,
                noise_ratio=sa.noise_ratio,
                slope=sa.slope,
                curvature=sa.curvature,
                regime=sa.regime.value,
                dt=sa.dt if hasattr(sa, "dt") else 1.0,
            )
            outcome = Outcome(
                kind="prediction",
                predicted_value=dto.predicted_value,
                confidence=dto.confidence_score,
                trend=dto.trend,
            )
            explanation = Explanation(
                series_id=dto.series_id,
                signal=signal,
                outcome=outcome,
                audit_trace_id=dto.audit_trace_id,
            )

            # Render metacognitive classifications
            rendered = self._renderer.render_structured_json(explanation)
            metacognitive = rendered.get("metacognitive", {})

            return {
                "structural_analysis": structural_dict,
                "metacognitive": metacognitive,
            }

        except Exception as exc:
            logger.warning(
                "enrichment_failed",
                extra={
                    "sensor_id": sensor_id,
                    "error": str(exc),
                },
            )
            return {}

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
