"""Ensemble de detectores de anomalías con voting ponderado.

Compone sub-detectores individuales (cada uno = una responsabilidad)
y una VotingStrategy desacoplada para combinar votos.

Arquitectura:
  SubDetector[] → votos individuales [0, 1]
  VotingStrategy → score final + decisión anomalía
  AnomalyNarrator → explicación legible

ISO 27001: Cada voto individual se registra en el resultado para
trazabilidad completa de la decisión.

.. versionchanged:: 2.0
    ``VotingAnomalyDetector`` now accepts optional ``sub_detectors``
    via constructor for dependency injection (MOD-2, ROB-2).
    ``create_default_detectors()`` extracts the default ensemble.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ....domain.entities.anomaly import AnomalyResult, AnomalySeverity
from ....domain.entities.sensor_reading import SensorWindow
from ....domain.ports.anomaly_detection_port import AnomalyDetectionPort

from .anomaly_config import AnomalyDetectorConfig
from .anomaly_narrator import build_anomaly_explanation
from .detector_protocol import SubDetector
from .detectors import (
    AccelerationZDetector,
    IQRDetector,
    IsolationForestDetector,
    LOFDetector,
    VelocityZDetector,
    ZScoreDetector,
)
from .detectors.isolation_forest_detector import IsolationForestNDDetector
from .detectors.lof_detector import LOFNDDetector
from .scoring_functions import compute_z_score
from .temporal_stats import TemporalTrainingStats
from .training_stats import TrainingStats
from .voting_strategy import VotingStrategy

logger = logging.getLogger(__name__)


def create_default_detectors(
    config: AnomalyDetectorConfig,
) -> List[SubDetector]:
    """Create the default ensemble of 8 sub-detectors.

    This factory function extracts the hardcoded detector list so it can
    be reused, overridden, or extended.  Pass the result to
    ``VotingAnomalyDetector(sub_detectors=...)`` to customize.

    Args:
        config: Anomaly detector configuration.

    Returns:
        List of 8 default sub-detectors.
    """
    return [
        ZScoreDetector(
            lower=config.z_vote_lower,
            upper=config.z_vote_upper,
        ),
        IQRDetector(),
        IsolationForestDetector(
            contamination=config.contamination,
            n_estimators=config.n_estimators,
            random_state=config.random_state,
        ),
        LOFDetector(
            contamination=config.contamination,
            max_neighbors=config.lof_max_neighbors,
        ),
        VelocityZDetector(
            lower=config.z_vote_lower,
            upper=config.z_vote_upper,
        ),
        AccelerationZDetector(
            lower=config.z_vote_lower,
            upper=config.z_vote_upper,
        ),
        IsolationForestNDDetector(
            contamination=config.contamination,
            n_estimators=config.n_estimators,
            random_state=config.random_state,
            min_training_points=config.min_training_points,
        ),
        LOFNDDetector(
            contamination=config.contamination,
            max_neighbors=config.lof_max_neighbors,
            min_training_points=config.min_training_points,
        ),
    ]


class VotingAnomalyDetector(AnomalyDetectionPort):
    """Ensemble de detectores de anomalías con voting.

    Compone sub-detectores individuales y delega la decisión
    a una VotingStrategy desacoplada.

    Attributes:
        _config: Configuración centralizada.
        _sub_detectors: Lista de sub-detectores individuales.
        _strategy: Estrategia de voting.
        _trained_flag: ``True`` si fue entrenado.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        voting_threshold: float = 0.5,
        weights: Optional[Dict[str, float]] = None,
        *,
        n_estimators: int = 100,
        random_state: int = 42,
        lof_max_neighbors: int = 20,
        min_training_points: int = 50,
        config: Optional[AnomalyDetectorConfig] = None,
        sub_detectors: Optional[List[SubDetector]] = None,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            self._config = AnomalyDetectorConfig(
                contamination=contamination,
                voting_threshold=voting_threshold,
                min_training_points=min_training_points,
                n_estimators=n_estimators,
                random_state=random_state,
                lof_max_neighbors=lof_max_neighbors,
                weights=weights or AnomalyDetectorConfig().weights,
            )

        if sub_detectors is not None:
            self._sub_detectors: List[SubDetector] = list(sub_detectors)
        else:
            self._sub_detectors = create_default_detectors(self._config)

        self._strategy = VotingStrategy(
            weights=self._config.weights,
            threshold=self._config.voting_threshold,
        )

        self._trained_flag: bool = False

        # Backward-compatible attributes for tests that inspect internals
        self._stats: TrainingStats = TrainingStats(
            mean=0.0, std=1e-9, q1=0.0, q3=0.0, iqr=0.0
        )
        self._temporal_stats: TemporalTrainingStats = TemporalTrainingStats.empty()

    @property
    def name(self) -> str:
        return "voting_anomaly_detector"

    def train(
        self,
        historical_values: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> None:
        """Entrena todos los sub-detectores con datos históricos.

        Args:
            historical_values: Serie temporal de entrenamiento.
            timestamps: Timestamps correspondientes (opcional).

        Raises:
            ValueError: Si no hay suficientes datos.
        """
        if len(historical_values) < self._config.min_training_points:
            raise ValueError(
                f"Se requieren >= {self._config.min_training_points} puntos para entrenar, "
                f"recibidos {len(historical_values)}"
            )

        kwargs: dict = {}
        if timestamps is not None and len(timestamps) == len(historical_values):
            kwargs["timestamps"] = timestamps

        for detector in self._sub_detectors:
            try:
                detector.train(historical_values, **kwargs)
            except Exception as exc:
                logger.warning(
                    "sub_detector_training_failed",
                    extra={"detector": detector.method_name, "error": str(exc)},
                )

        self._trained_flag = True

        # Update backward-compatible stats attributes
        from .training_stats import compute_training_stats
        from .temporal_stats import compute_temporal_training_stats

        self._stats = compute_training_stats(historical_values)
        if "timestamps" in kwargs:
            self._temporal_stats = compute_temporal_training_stats(
                historical_values, kwargs["timestamps"]
            )
        else:
            self._temporal_stats = TemporalTrainingStats.empty()

        logger.info(
            "voting_detector_trained",
            extra={
                "n_points": len(historical_values),
                "mean": round(self._stats.mean, 4),
                "std": round(self._stats.std, 4),
                "temporal_trained": self._temporal_stats.has_temporal,
                "n_sub_detectors": len(self._sub_detectors),
            },
        )

    def detect(self, window: SensorWindow) -> AnomalyResult:
        """Detecta anomalías usando voting ponderado de sub-detectores.

        Args:
            window: Ventana temporal del sensor.

        Returns:
            ``AnomalyResult`` con score combinado y votos individuales.
        """
        if not self._trained_flag:
            raise RuntimeError("Detector no entrenado")

        if window.is_empty or window.last_value is None:
            return AnomalyResult.normal(series_id=str(window.sensor_id))

        value = window.last_value

        # Build context for sub-detectors
        vote_kwargs = self._build_vote_context(window)

        # Collect votes from all sub-detectors
        votes: Dict[str, float] = {}
        for detector in self._sub_detectors:
            if not detector.is_trained:
                continue
            try:
                v = detector.vote(value, **vote_kwargs)
                if v is not None:
                    votes[detector.method_name] = v
            except Exception as exc:
                logger.debug(
                    "sub_detector_vote_failed",
                    extra={"detector": detector.method_name, "error": str(exc)},
                )

        # Combine votes via strategy
        final_score = self._strategy.combine(votes)
        is_anomaly = self._strategy.is_anomaly(final_score)
        confidence = self._strategy.confidence(votes)
        severity = AnomalySeverity.from_score(final_score)

        # Compute z-scores for narrator
        z = compute_z_score(value, self._stats.mean, self._stats.std)
        vel_z = self._extract_vel_z(window)
        acc_z = self._extract_acc_z(window)

        explanation = build_anomaly_explanation(
            votes, z_score=z, vel_z_score=vel_z, acc_z_score=acc_z,
        )

        logger.debug(
            "voting_detection",
            extra={
                "series_id": str(window.sensor_id),
                "value": value,
                "votes": {k: round(v, 3) for k, v in votes.items()},
                "final_score": round(final_score, 3),
                "is_anomaly": is_anomaly,
                "temporal_active": self._temporal_stats.has_temporal,
            },
        )

        return AnomalyResult(
            series_id=str(window.sensor_id),
            is_anomaly=is_anomaly,
            score=final_score,
            method_votes=votes,
            confidence=confidence,
            explanation=explanation,
            severity=severity,
        )

    def is_trained(self) -> bool:
        return self._trained_flag

    # --- Private helpers ---

    def _build_vote_context(self, window: SensorWindow) -> dict:
        """Builds kwargs context for sub-detector vote() calls."""
        ctx: dict = {}

        if self._temporal_stats.has_temporal and window.size >= 2:
            try:
                ctx["temporal_features"] = window.temporal_features
            except Exception:
                pass

        if self._temporal_stats.has_temporal and window.size >= 3:
            try:
                import numpy as np
                tf = window.temporal_features
                if tf.has_velocity and tf.has_acceleration:
                    ctx["nd_features"] = np.array([[
                        window.last_value,
                        tf.last_velocity,
                        tf.last_acceleration,
                    ]])
            except Exception:
                pass

        return ctx

    def _extract_vel_z(self, window: SensorWindow) -> float:
        if not self._temporal_stats.has_temporal or window.size < 2:
            return 0.0
        try:
            tf = window.temporal_features
            if tf.has_velocity:
                return compute_z_score(
                    tf.last_velocity,
                    self._temporal_stats.vel_mean,
                    self._temporal_stats.vel_std,
                )
        except Exception:
            pass
        return 0.0

    def _extract_acc_z(self, window: SensorWindow) -> float:
        if not self._temporal_stats.has_temporal or window.size < 3:
            return 0.0
        try:
            tf = window.temporal_features
            if tf.has_acceleration:
                return compute_z_score(
                    tf.last_acceleration,
                    self._temporal_stats.acc_mean,
                    self._temporal_stats.acc_std,
                )
        except Exception:
            pass
        return 0.0
