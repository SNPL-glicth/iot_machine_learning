"""Helper para tracking de predicciones en MLflow.

Separa la lógica de tracking del PredictionService para mantenerlo ≤180 líneas.
Este helper actúa como decorator/bridge entre el use case y el tracker.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from iot_machine_learning.domain.ports.experiment_tracker_port import (
    ExperimentTrackerPort,
    NullExperimentTracker,
)
from iot_machine_learning.application.dto.prediction_dto import PredictionDTO

logger = logging.getLogger(__name__)


class PredictionTrackerHelper:
    """Helper que instrumenta predicciones para MLflow tracking.

    Se usa desde PredictionService sin aumentar su conteo de líneas.
    Toda la lógica de tracking vive aquí.

    Attributes:
        _tracker: Port de experiment tracking (MLflow o Null).
        _prediction_count: Contador de predicciones para steps.
    """

    def __init__(self, tracker: Optional[ExperimentTrackerPort] = None) -> None:
        """Inicializa con tracker opcional.

        Args:
            tracker: Tracker port. Si None, usa NullExperimentTracker (no-op).
        """
        self._tracker = tracker or NullExperimentTracker()
        self._prediction_count = 0
        self._plasticity_update_count = 0

    def track_prediction(
        self,
        sensor_id: int,
        dto: PredictionDTO,
        regime: Optional[str] = None,
        window_size: int = 60,
        elapsed_ms: float = 0.0,
    ) -> None:
        """Loguea una predicción en MLflow.

        Args:
            sensor_id: ID del sensor.
            dto: DTO de predicción con valores y metadatos.
            regime: Régimen actual (STABLE, TRENDING, VOLATILE, etc.).
            window_size: Tamaño de ventana usado.
            elapsed_ms: Tiempo de procesamiento en ms.
        """
        self._prediction_count += 1

        try:
            # Métricas
            metrics: Dict[str, float] = {
                "confidence_score": dto.confidence_score,
                "elapsed_ms": elapsed_ms,
            }
            if dto.confidence_interval:
                lower, upper = dto.confidence_interval
                metrics["confidence_interval_width"] = upper - lower

            # Parámetros
            params = {
                "engine_name": dto.engine_name,
                "regime": regime or "unknown",
                "series_id": dto.series_id,
                "window_size": window_size,
            }

            # Tags
            tags = {
                "pipeline_version": "0.2.1-GOLD",
                "sensor_id": str(sensor_id),
            }

            self._tracker.log_metrics(metrics, step=self._prediction_count)
            self._tracker.log_params(params)
            self._tracker.set_tags(tags)

            logger.debug(
                "prediction_tracked",
                extra={
                    "sensor_id": sensor_id,
                    "step": self._prediction_count,
                    "engine": dto.engine_name,
                },
            )

        except Exception as exc:
            # Fail-safe: nunca propagar excepción de tracking
            logger.debug("prediction_track_failed", extra={"error": str(exc)})

    def track_plasticity_update(
        self,
        regime: str,
        engine_weights: Dict[str, float],
    ) -> None:
        """Loguea actualización de plasticity weights en MLflow.

        Args:
            regime: Régimen actual.
            engine_weights: Dict {engine_name: weight}.
        """
        self._plasticity_update_count += 1

        try:
            # Loguear cada peso como métrica separada
            metrics = {
                f"engine_weight_{name}": weight
                for name, weight in engine_weights.items()
            }
            # Añadir entropía de pesos como métrica de diversidad
            if engine_weights:
                weights_list = list(engine_weights.values())
                total = sum(weights_list)
                if total > 0:
                    import math

                    probs = [w / total for w in weights_list]
                    entropy = -sum(p * math.log(p + 1e-10) for p in probs)
                    metrics["weight_entropy"] = entropy

            self._tracker.log_metrics(metrics, step=self._plasticity_update_count)

            # Loguear régimen como parámetro
            self._tracker.log_param("plasticity_regime", regime)

            logger.debug(
                "plasticity_tracked",
                extra={
                    "regime": regime,
                    "step": self._plasticity_update_count,
                    "engines": list(engine_weights.keys()),
                },
            )

        except Exception as exc:
            logger.debug("plasticity_track_failed", extra={"error": str(exc)})

    def track_anomaly(
        self,
        anomaly_score: float,
        severity: str,
        detector_count: int,
        voting_threshold: float,
    ) -> None:
        """Loguea detección de anomalía en MLflow.

        Args:
            anomaly_score: Score de anomalía [0, 1].
            severity: Nivel de severidad (none, low, medium, high, critical).
            detector_count: Número de detectores que votaron.
            voting_threshold: Umbral de voting.
        """
        try:
            metrics = {
                "anomaly_score": anomaly_score,
                "severity_level": self._severity_to_numeric(severity),
            }

            params = {
                "detector_count": detector_count,
                "voting_threshold": voting_threshold,
            }

            self._tracker.log_metrics(metrics)
            self._tracker.log_params(params)

            logger.debug(
                "anomaly_tracked",
                extra={
                    "score": anomaly_score,
                    "severity": severity,
                },
            )

        except Exception as exc:
            logger.debug("anomaly_track_failed", extra={"error": str(exc)})

    @staticmethod
    def _severity_to_numeric(severity: str) -> int:
        """Convierte severidad a valor numérico para MLflow."""
        mapping = {
            "none": 0,
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 4,
        }
        return mapping.get(severity.lower(), -1)

    def start_run(self, run_name: Optional[str] = None) -> str:
        """Inicia run en el tracker."""
        return self._tracker.start_run(
            run_name=run_name,
            tags={"pipeline_version": "0.2.1-GOLD"},
        )

    def end_run(self, status: Optional[str] = None) -> None:
        """Finaliza run en el tracker."""
        self._tracker.end_run(status)

    @property
    def tracker(self) -> ExperimentTrackerPort:
        """Acceso al tracker subyacente."""
        return self._tracker
