"""Servicio de predicción por regresión — capa Modeling.

Extraído de sensor_processor.py._model_prediction() y _fallback_prediction().
Responsabilidad ÚNICA: dado un modelo de regresión y datos de serie,
calcular valor predicho, trend, confianza, anomalía y score.

No contiene I/O (no lee BD, no persiste).
No genera explicaciones (eso es Narrative).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from iot_machine_learning.ml_service.config.ml_config import RegressionConfig
    from iot_machine_learning.ml_service.models.regression_model import RegressionModel
    from iot_machine_learning.ml_service.repository.sensor_repository import SensorSeries
    from iot_machine_learning.ml_service.trainers.isolation_trainer import IsolationForestTrainer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionResult:
    """Resultado de predicción por regresión.

    Value object puro — sin lógica de negocio ni I/O.
    """

    predicted_value: float
    trend: str
    confidence: float
    anomaly: bool
    anomaly_score: float
    window_points_effective: int


class RegressionPredictionService:
    """Calcula predicciones usando regresión lineal + IsolationForest.

    Responsabilidad ÚNICA: Modeling (cálculo matemático).
    No lee BD. No persiste. No genera explicaciones.
    """

    def predict_with_model(
        self,
        series: "SensorSeries",
        reg_model: "RegressionModel",
        last_minutes: float,
        reg_cfg: "RegressionConfig",
        iso_trainer: "IsolationForestTrainer",
        sensor_id: int,
    ) -> PredictionResult:
        """Predicción con modelo de regresión entrenado.

        Args:
            series: Serie temporal del sensor.
            reg_model: Modelo de regresión entrenado.
            last_minutes: Minutos del último punto.
            reg_cfg: Configuración de regresión.
            iso_trainer: Trainer de IsolationForest.
            sensor_id: ID del sensor.

        Returns:
            ``PredictionResult`` con valor, trend, confianza y anomalía.
        """
        from iot_machine_learning.ml_service.trainers.regression_trainer import (
            predict_future_value_clamped,
        )
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

        return PredictionResult(
            predicted_value=predicted_value,
            trend=trend,
            confidence=confidence,
            anomaly=anomaly,
            anomaly_score=anomaly_score,
            window_points_effective=window_points_effective,
        )

    def predict_fallback(
        self,
        values: list[float],
        reg_cfg: "RegressionConfig",
    ) -> PredictionResult:
        """Predicción fallback cuando no hay modelo (promedio simple).

        Args:
            values: Valores recientes del sensor.
            reg_cfg: Configuración de regresión.

        Returns:
            ``PredictionResult`` con promedio y confianza mínima.
        """
        last_values = values[-min(5, len(values)):]
        predicted_value = float(sum(last_values) / len(last_values))

        return PredictionResult(
            predicted_value=predicted_value,
            trend="stable",
            confidence=reg_cfg.min_confidence,
            anomaly=False,
            anomaly_score=0.0,
            window_points_effective=len(values),
        )
