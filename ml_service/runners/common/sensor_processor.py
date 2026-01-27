"""Procesador de sensor individual.

Responsabilidad única: Procesar un sensor y generar predicción.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy.engine import Connection

try:
    from .model_manager import ModelManager
    from .prediction_writer import PredictionWriter
    from .event_writer import EventWriter
    from .severity_classifier import SeverityClassifier
except ImportError:
    from model_manager import ModelManager
    from prediction_writer import PredictionWriter
    from event_writer import EventWriter
    from severity_classifier import SeverityClassifier

if TYPE_CHECKING:
    from iot_machine_learning.ml_service.config.ml_config import GlobalMLConfig
    from iot_machine_learning.ml_service.trainers.isolation_trainer import IsolationForestTrainer

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SensorProcessor:
    """Procesa un sensor individual y genera predicción.
    
    Orquesta:
    - Carga de datos del sensor
    - Entrenamiento de regresión
    - Detección de anomalías
    - Construcción de explicación
    - Persistencia de predicción y eventos
    """
    
    def __init__(self):
        self._model_manager = ModelManager()
        self._severity_classifier = SeverityClassifier()
        self._event_writer = EventWriter()
        self._prediction_writer = PredictionWriter(self._event_writer)
    
    def process(
        self,
        conn: Connection,
        sensor_id: int,
        ml_cfg: "GlobalMLConfig",
        iso_trainer: "IsolationForestTrainer",
    ) -> None:
        """Procesa un sensor y genera predicción.
        
        Args:
            conn: Conexión a BD
            sensor_id: ID del sensor a procesar
            ml_cfg: Configuración ML global
            iso_trainer: Trainer de IsolationForest
        """
        # Imports locales para evitar circular imports
        from iot_machine_learning.ml_service.repository.sensor_repository import (
            load_sensor_series,
            load_sensor_metadata,
            get_device_id_for_sensor,
        )
        from iot_machine_learning.ml_service.trainers.regression_trainer import (
            train_regression_for_sensor,
            predict_future_value_clamped,
        )
        from iot_machine_learning.ml_service.models.regression_model import compute_trend
        
        # 1. Cargar datos del sensor
        series = load_sensor_series(
            conn, sensor_id, limit_points=ml_cfg.regression.window_points
        )
        sensor_meta = load_sensor_metadata(conn, sensor_id)
        
        if not series.values:
            logger.info("[SENSOR_PROC] sensor=%s sin datos, omitiendo", sensor_id)
            return
        
        n_points = len(series.values)
        logger.info(
            "[SENSOR_PROC] sensor=%s: %d lecturas cargadas",
            sensor_id, n_points
        )
        
        reg_cfg = ml_cfg.regression
        
        # 2. Entrenar regresión
        reg_model, last_minutes = train_regression_for_sensor(conn, sensor_id, reg_cfg)
        
        if reg_model is None or last_minutes is None:
            # Fallback a promedio simple
            predicted_value, trend, confidence, anomaly, anomaly_score, window_points_effective = \
                self._fallback_prediction(series.values, reg_cfg)
        else:
            # Predicción con modelo
            predicted_value, trend, confidence, anomaly, anomaly_score, window_points_effective = \
                self._model_prediction(
                    series, reg_model, last_minutes, reg_cfg, iso_trainer, sensor_id
                )
        
        # 3. Subordinar anomalía a umbrales del usuario
        if anomaly and self._severity_classifier.is_value_within_user_thresholds(
            conn, sensor_id, predicted_value
        ):
            logger.debug(
                "[SENSOR_PROC] anomaly suppressed sensor=%s (within user thresholds)",
                sensor_id
            )
            anomaly = False
        
        # 4. Construir explicación
        user_range = self._severity_classifier.get_user_defined_range(conn, sensor_id)
        explanation = self._build_explanation(
            sensor_meta=sensor_meta,
            predicted_value=predicted_value,
            trend=trend,
            anomaly=anomaly,
            anomaly_score=anomaly_score,
            confidence=confidence,
            horizon_minutes=reg_cfg.horizon_minutes,
            user_defined_range=user_range,
        )
        
        # 5. Persistir predicción
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
            window_points=window_points_effective,
        )
        
        logger.info(
            "[SENSOR_PROC] prediction id=%s sensor=%s pred=%.4f conf=%.2f",
            prediction_id, sensor_id, explanation.predicted_value, explanation.confidence
        )
        
        # 6. Generar eventos
        self._event_writer.insert_threshold_event_if_needed(
            conn,
            sensor_id=sensor_id,
            device_id=device_id,
            prediction_id=prediction_id,
            predicted_value=predicted_value,
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
            sensor_id, predicted_value, trend, explanation.anomaly, explanation.anomaly_score
        )
    
    def _fallback_prediction(self, values: list, reg_cfg) -> tuple:
        """Predicción fallback cuando no hay modelo."""
        last_values = values[-min(5, len(values)):]
        predicted_value = float(sum(last_values) / len(last_values))
        trend = "stable"
        confidence = reg_cfg.min_confidence
        anomaly = False
        anomaly_score = 0.0
        window_points_effective = len(values)
        return predicted_value, trend, confidence, anomaly, anomaly_score, window_points_effective
    
    def _model_prediction(
        self,
        series,
        reg_model,
        last_minutes: float,
        reg_cfg,
        iso_trainer: "IsolationForestTrainer",
        sensor_id: int,
    ) -> tuple:
        """Predicción con modelo de regresión."""
        from iot_machine_learning.ml_service.trainers.regression_trainer import predict_future_value_clamped
        from iot_machine_learning.ml_service.models.regression_model import compute_trend
        
        series_min = float(min(series.values))
        series_max = float(max(series.values))
        last_value = float(series.values[-1])
        
        predicted_value = predict_future_value_clamped(
            reg_model,
            last_minutes,
            last_value=last_value,
            series_min=series_min,
            series_max=series_max,
            max_change_ratio=0.5,
        )
        trend = compute_trend(reg_model.coef_)
        
        # Residuales para IsolationForest
        t0 = series.timestamps[0]
        xs = [[((ts - t0).total_seconds() / 60.0)] for ts in series.timestamps]
        X = np.asarray(xs, dtype=float)
        y = np.asarray(series.values, dtype=float)
        y_hat_hist = reg_model.intercept_ + reg_model.coef_ * X.ravel()
        residuals = y - y_hat_hist
        
        window_points_effective = len(series.values)
        
        # Confianza basada en R² + nº de puntos
        conf_r2 = max(0.0, min(1.0, reg_model.r2))
        raw_conf = min(1.0, window_points_effective / reg_cfg.window_points)
        confidence = max(
            reg_cfg.min_confidence,
            min(reg_cfg.max_confidence, 0.5 * (conf_r2 + raw_conf)),
        )
        
        # Anomalía con IsolationForest
        anomaly = False
        anomaly_score = 0.0
        model = iso_trainer.fit_for_sensor(sensor_id, residuals)
        if model is not None:
            last_residual = float(residuals[-1])
            anomaly_score, anomaly = iso_trainer.score_new_point(sensor_id, last_residual)
        
        return predicted_value, trend, confidence, anomaly, anomaly_score, window_points_effective
    
    def _build_explanation(
        self,
        *,
        sensor_meta,
        predicted_value: float,
        trend: str,
        anomaly: bool,
        anomaly_score: float,
        confidence: float,
        horizon_minutes: int,
        user_defined_range: tuple[float, float] | None = None,
    ):
        """Construye explicación de la predicción."""
        from iot_machine_learning.ml_service.explain.explanation_builder import PredictionExplanation
        
        result = self._severity_classifier.classify(
            sensor_type=sensor_meta.sensor_type,
            location=sensor_meta.location,
            predicted_value=predicted_value,
            trend=trend,
            anomaly=anomaly,
            anomaly_score=anomaly_score,
            confidence=confidence,
            horizon_minutes=horizon_minutes,
            user_defined_range=user_defined_range,
        )
        
        # Evitar contradicciones
        if result.severity == "critical" and anomaly_score <= 0:
            anomaly_score = 0.5
        
        # Construir mensaje
        if result.severity == "critical":
            short_message = f"Riesgo crítico previsto en {sensor_meta.sensor_type} en {sensor_meta.location}."
        elif result.severity == "warning":
            short_message = f"Comportamiento inusual previsto en {sensor_meta.sensor_type} en {sensor_meta.location}."
        else:
            short_message = f"Predicción estable para {sensor_meta.sensor_type} en {sensor_meta.location}."
        
        details = {
            "predicted_value": float(predicted_value),
            "trend": trend,
            "anomaly_score": float(anomaly_score),
            "confidence": float(confidence),
            "horizon_minutes": int(horizon_minutes),
            "risk_level": result.risk_level,
            "sensor_type": sensor_meta.sensor_type,
            "location": sensor_meta.location,
        }
        
        explanation_payload = {
            "source": "ml_baseline",
            "severity": result.severity.upper(),
            "short_message": short_message,
            "recommended_action": result.recommended_action,
            "details": details,
        }
        explanation_json = json.dumps(explanation_payload, ensure_ascii=False)
        
        return PredictionExplanation(
            sensor_id=sensor_meta.sensor_id,
            predicted_value=predicted_value,
            trend=trend,
            anomaly=anomaly,
            anomaly_score=anomaly_score,
            confidence=confidence,
            explanation=explanation_json,
            risk_level=result.risk_level,
            severity=result.severity,
            action_required=result.action_required,
            recommended_action=result.recommended_action,
        )
