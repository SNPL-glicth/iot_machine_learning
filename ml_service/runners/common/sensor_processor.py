"""Procesador de sensor individual.

REFACTORIZADO: Thin orchestrator que delega a servicios especializados.

Antes (God Object — 5 responsabilidades):
  - Carga de datos (Percepción)
  - Regresión + residuales (Modelado)
  - Anomalía IsolationForest (Decisión)
  - Construcción de explicación (Narrativa)
  - Persistencia de predicción y eventos (Infra)

Ahora (Orchestrator — 1 responsabilidad):
  - Coordina el flujo delegando a:
    • RegressionPredictionService (Modeling)
    • PredictionNarrator (Narrative)
    • ModelManager, PredictionWriter, EventWriter (Infra — ya existían)

Módulos extraídos:
  - regression_prediction_service.py (Modeling puro)
  - prediction_narrator.py (Narrative puro)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Optional

from sqlalchemy.engine import Connection

try:
    from .model_manager import ModelManager
    from .prediction_writer import PredictionWriter
    from .event_writer import EventWriter
    from iot_machine_learning.infrastructure.ml.cognitive.severity_classifier import SeverityClassifier
    from .regression_prediction_service import RegressionPredictionService
    from .prediction_narrator import PredictionNarrator
except ImportError:
    from model_manager import ModelManager
    from prediction_writer import PredictionWriter
    from event_writer import EventWriter
    from iot_machine_learning.infrastructure.ml.cognitive.severity_classifier import SeverityClassifier
    from regression_prediction_service import RegressionPredictionService
    from prediction_narrator import PredictionNarrator

if TYPE_CHECKING:
    from iot_machine_learning.ml_service.config.ml_config import GlobalMLConfig
    from iot_machine_learning.ml_service.trainers.isolation_trainer import IsolationForestTrainer
    from iot_machine_learning.ml_service.runners.adapters.enterprise_prediction import (
        EnterprisePredictionAdapter,
    )

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SensorProcessor:
    """Procesa un sensor individual y genera predicción.

    Thin orchestrator: coordina servicios especializados.
    No contiene cálculos, reglas de negocio ni generación de texto.
    """

    def __init__(self):
        self._model_manager = ModelManager()
        self._severity_classifier = SeverityClassifier()
        self._event_writer = EventWriter()
        self._prediction_writer = PredictionWriter(self._event_writer)
        self._regression_service = RegressionPredictionService()
        self._narrator = PredictionNarrator(self._severity_classifier)

    def process(
        self,
        conn: Connection,
        sensor_id: int,
        ml_cfg: "GlobalMLConfig",
        iso_trainer: "IsolationForestTrainer",
        enterprise_adapter: Optional["EnterprisePredictionAdapter"] = None,
    ) -> None:
        """Procesa un sensor y genera predicción.

        Args:
            conn: Conexión a BD
            sensor_id: ID del sensor a procesar
            ml_cfg: Configuración ML global
            iso_trainer: Trainer de IsolationForest
            enterprise_adapter: Adapter enterprise (opcional, controlado por flags)
        """
        from iot_machine_learning.ml_service.repository.sensor_repository import (
            load_sensor_series,
            load_sensor_metadata,
            get_device_id_for_sensor,
        )
        from iot_machine_learning.ml_service.trainers.regression_trainer import (
            train_regression_for_sensor,
        )

        # 1. Cargar datos del sensor (Percepción — delegado a repository)
        series = load_sensor_series(
            conn, sensor_id, limit_points=ml_cfg.regression.window_points
        )
        sensor_meta = load_sensor_metadata(conn, sensor_id)

        if not series.values:
            logger.info("[SENSOR_PROC] sensor=%s sin datos, omitiendo", sensor_id)
            return

        logger.info(
            "[SENSOR_PROC] sensor=%s: %d lecturas cargadas",
            sensor_id, len(series.values)
        )

        reg_cfg = ml_cfg.regression

        # 2. Calcular predicción
        # 2a. Ruta enterprise (si adapter disponible)
        if enterprise_adapter is not None:
            enterprise_result = enterprise_adapter.predict(
                sensor_id=sensor_id,
                window_size=reg_cfg.window_points,
            )
            logger.info(
                "[SENSOR_PROC] enterprise sensor=%s engine=%s conf=%.2f",
                sensor_id, enterprise_result.engine_used, enterprise_result.confidence,
            )
            from .regression_prediction_service import PredictionResult as _PR
            pred_result = _PR(
                predicted_value=enterprise_result.predicted_value,
                trend=enterprise_result.trend,
                confidence=enterprise_result.confidence,
                anomaly=enterprise_result.anomaly_score > 0.5,
                anomaly_score=enterprise_result.anomaly_score,
                window_points_effective=reg_cfg.window_points,
            )
        else:
            # 2b. Ruta legacy (Modeling — delegado a RegressionPredictionService)
            reg_model, last_minutes = train_regression_for_sensor(conn, sensor_id, reg_cfg)

            if reg_model is None or last_minutes is None:
                pred_result = self._regression_service.predict_fallback(
                    series.values, reg_cfg
                )
            else:
                pred_result = self._regression_service.predict_with_model(
                    series, reg_model, last_minutes, reg_cfg, iso_trainer, sensor_id
                )

        # 3. Subordinar anomalía a umbrales del usuario (Decisión)
        anomaly = pred_result.anomaly
        if anomaly and self._severity_classifier.is_value_within_user_thresholds(
            conn, sensor_id, pred_result.predicted_value
        ):
            logger.debug(
                "[SENSOR_PROC] anomaly suppressed sensor=%s (within user thresholds)",
                sensor_id
            )
            anomaly = False

        # 4. Construir explicación (Narrative — delegado a PredictionNarrator)
        user_range = self._severity_classifier.get_user_defined_range(conn, sensor_id)
        explanation = self._narrator.build_explanation(
            sensor_meta=sensor_meta,
            predicted_value=pred_result.predicted_value,
            trend=pred_result.trend,
            anomaly=anomaly,
            anomaly_score=pred_result.anomaly_score,
            confidence=pred_result.confidence,
            horizon_minutes=reg_cfg.horizon_minutes,
            user_defined_range=user_range,
        )

        # 5. Persistir predicción (Infra — delegado a writers existentes)
        device_id = get_device_id_for_sensor(conn, sensor_id)
        model_id = self._model_manager.get_or_create_model_id(conn, sensor_id)
        target_ts = _utc_now() + timedelta(minutes=reg_cfg.horizon_minutes)

        prediction_id = self._prediction_writer.insert_prediction(
            conn,
            model_id=model_id,
            sensor_id=sensor_id,
            device_id=device_id,
            explanation=explanation,
            target_ts_utc=target_ts,
            horizon_minutes=reg_cfg.horizon_minutes,
            window_points=pred_result.window_points_effective,
        )

        logger.info(
            "[SENSOR_PROC] prediction id=%s sensor=%s pred=%.4f conf=%.2f",
            prediction_id, sensor_id, explanation.predicted_value, explanation.confidence
        )

        # 6. Generar eventos (Infra — delegado a EventWriter existente)
        self._event_writer.insert_threshold_event_if_needed(
            conn,
            sensor_id=sensor_id,
            device_id=device_id,
            prediction_id=prediction_id,
            predicted_value=pred_result.predicted_value,
        )

        self._event_writer.insert_anomaly_event(
            conn,
            sensor_id=sensor_id,
            device_id=device_id,
            prediction_id=prediction_id,
            explanation=explanation,
            severity_classifier=self._severity_classifier,
        )

        logger.info(
            "[SENSOR_PROC] OK sensor=%s pred=%.3f trend=%s anomaly=%s score=%.4f",
            sensor_id, pred_result.predicted_value, pred_result.trend,
            explanation.anomaly, explanation.anomaly_score
        )
