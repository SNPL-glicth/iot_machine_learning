"""Ensemble de detectores de anomalías con voting ponderado.
Compone sub-detectores individuales y delega la decisión
a una VotingStrategy desacoplada.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.preprocessing import RobustScaler

from iot_machine_learning.domain.entities.anomaly import AnomalyResult
from iot_machine_learning.domain.entities.sensor_reading import SensorWindow
from iot_machine_learning.domain.ports.anomaly_detection_port import AnomalyDetectionPort
from iot_machine_learning.domain.policies.threshold_policy import ThresholdPolicy

from .config import AnomalyDetectorConfig
from .protocol import SubDetector
from ..factory import create_default_detectors
from ..narration import build_anomaly_explanation
from ..scoring import compute_z_score, TrainingStats, TemporalTrainingStats
from ..voting import build_vote_context, extract_acc_z, extract_vel_z, VotingStrategy

logger = logging.getLogger(__name__)


class VotingAnomalyDetector(AnomalyDetectionPort):

    @property
    def name(self) -> str:
        return "voting_anomaly_detector"

    def __init__(
        self,
        config: Optional[AnomalyDetectorConfig] = None,
        sub_detectors: Optional[List[SubDetector]] = None,
        series_id: Optional[str] = None,
        enable_adaptive_weights: bool = False,
        **kwargs: object,
    ) -> None:
        self._config = config or AnomalyDetectorConfig()
        self._sub_detectors = (
            sub_detectors
            if sub_detectors is not None
            else create_default_detectors(self._config)
        )
        self._series_id = series_id
        self._strategy = VotingStrategy(
            weights=self._config.weights,
            threshold=self._config.voting_threshold,
        )
        self._trained_flag = False
        self._scaler = None
        self._stats = TrainingStats(mean=0.0, std=1e-9, q1=0.0, q3=0.0, iqr=0.0)
        self._temporal_stats = TemporalTrainingStats.empty()

    def train(self, historical_values, timestamps=None):
        if len(historical_values) < self._config.min_training_points:
            raise ValueError(
                f"Need >= {self._config.min_training_points} points, got {len(historical_values)}"
            )

        try:
            self._scaler = RobustScaler()
            arr = np.array(historical_values).reshape(-1, 1)
            values_norm = self._scaler.fit_transform(arr).flatten().tolist()
        except Exception as exc:
            logger.warning("normalization_failed", extra={"error": str(exc)})
            values_norm = historical_values
            self._scaler = None

        kwargs = {}
        if timestamps is not None and len(timestamps) == len(historical_values):
            kwargs["timestamps"] = timestamps

        for detector in self._sub_detectors:
            try:
                detector.train(values_norm, **kwargs)
            except Exception as exc:
                logger.warning(
                    "sub_detector_training_failed",
                    extra={"detector": detector.method_name, "error": str(exc)},
                )

        self._trained_flag = True

        from ..scoring.training import compute_training_stats
        from ..scoring.temporal import compute_temporal_training_stats

        self._stats = compute_training_stats(historical_values)
        if "timestamps" in kwargs:
            self._temporal_stats = compute_temporal_training_stats(
                historical_values, kwargs["timestamps"]
            )
        else:
            self._temporal_stats = TemporalTrainingStats.empty()

        logger.info(
            "voting_detector_trained",
            extra={"n_points": len(historical_values), "n_detectors": len(self._sub_detectors)},
        )

    def detect(self, window: SensorWindow) -> AnomalyResult:
        if not self._trained_flag:
            if window.size >= self._config.min_training_points:
                self.train(window.values, timestamps=window.timestamps)
            else:
                return AnomalyResult(
                    series_id=str(window.series_id),
                    is_anomaly=False,
                    score=0.0,
                    confidence=0.0,
                    method_votes={"cold_start": 0.0},
                    explanation="Cold start: insufficient data",
                    context={"reason": "auto_train_skipped", "n": window.size},
                )

        if window.is_empty or window.last_value is None:
            return AnomalyResult.normal(series_id=str(window.sensor_id))

        value = window.last_value
        if self._scaler is not None:
            try:
                value = self._scaler.transform(np.array([[value]])).flatten()[0]
            except Exception as exc:
                logger.warning("scaler_transform_failed", extra={"error": str(exc)})

        vote_kwargs = build_vote_context(window, self._temporal_stats)
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

        final_score = self._strategy.combine(votes)
        is_anomaly = self._strategy.is_anomaly(final_score)
        confidence = self._strategy.confidence(votes)
        severity = ThresholdPolicy.default().classify_score(final_score)

        z = compute_z_score(value, self._stats.mean, self._stats.std)
        vel_z = extract_vel_z(window, self._temporal_stats)
        acc_z = extract_acc_z(window, self._temporal_stats)
        explanation = build_anomaly_explanation(
            votes, z_score=z, vel_z_score=vel_z, acc_z_score=acc_z,
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
