"""Ensemble de detectores de anomalías con voting ponderado.
Compone sub-detectores individuales y delega la decisión
a una VotingStrategy desacoplada.
"""
from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from sklearn.preprocessing import RobustScaler

from iot_machine_learning.domain.entities.iot.sensor_reading import SensorWindow
from iot_machine_learning.domain.entities.results.anomaly import AnomalyResult
from iot_machine_learning.domain.policies.threshold_policy import ThresholdPolicy
from iot_machine_learning.domain.ports.anomaly_detection_port import AnomalyDetectionPort

from ..factory import create_default_detectors
from ..narration import build_anomaly_explanation
from ..scoring import TemporalTrainingStats, TrainingStats, compute_z_score
from ..voting import VotingStrategy, build_vote_context, extract_acc_z, extract_vel_z
from .config import AnomalyDetectorConfig
from .protocol import SubDetector

logger = logging.getLogger(__name__)


class VotingAnomalyDetector(AnomalyDetectionPort):

    @property
    def name(self) -> str:
        return "voting_anomaly_detector"

    def __init__(
        self,
        config: AnomalyDetectorConfig | None = None,
        sub_detectors: list[SubDetector] | None = None,
        series_id: str | None = None,
        enable_adaptive_weights: bool = False,
        calibration_layer: Any | None = None,
        **kwargs: object,
    ) -> None:
        if config is None:
            cfg_fields = set(getattr(AnomalyDetectorConfig, "__dataclass_fields__", {}).keys())
            cfg_kwargs = {k: v for k, v in kwargs.items() if k in cfg_fields}
            self._config = AnomalyDetectorConfig(**cfg_kwargs) if cfg_kwargs else AnomalyDetectorConfig()  # type: ignore[arg-type]
        else:
            self._config = config
        self._sub_detectors = (
            list(sub_detectors) if sub_detectors is not None else create_default_detectors(self._config)
        )
        self._series_id = series_id
        self._strategy = VotingStrategy(
            weights=self._config.weights, threshold=self._config.voting_threshold,
        )
        self._trained_flag: bool = False
        self._scaler: RobustScaler | None = None
        self._stats = TrainingStats(mean=0.0, std=1e-9, q1=0.0, q3=0.0, iqr=0.0)
        self._temporal_stats = TemporalTrainingStats.empty()
        self._calibration_layer = calibration_layer

    def train(
        self,
        historical_values: list[float],
        timestamps: list[float] | None = None,
    ) -> None:
        if len(historical_values) < self._config.min_training_points:
            raise ValueError(
                f"Need >= {self._config.min_training_points} points, got {len(historical_values)}"
            )

        try:
            self._scaler = RobustScaler()
            arr = np.asarray(historical_values, dtype=np.float64).reshape(-1, 1)
            values_norm: list[float] = [
                float(x) for x in self._scaler.fit_transform(cast(Any, arr)).flatten()
            ]
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

        from ..scoring.temporal import compute_temporal_training_stats
        from ..scoring.training import compute_training_stats

        self._stats = compute_training_stats(historical_values)
        if "timestamps" in kwargs:
            self._temporal_stats = compute_temporal_training_stats(
                historical_values, kwargs["timestamps"]
            )
        else:
            self._temporal_stats = TemporalTrainingStats.empty()

        if self._calibration_layer is not None:
            for det in self._sub_detectors:
                if det.is_trained:
                    is_inv = det.method_name in ("isolation_forest", "local_outlier_factor", "lof_temporal")
                    raw_sc = det.get_training_raw_scores(values_norm, **kwargs)
                    self._calibration_layer.calibrate_detector(det.method_name, raw_sc, is_inverted=is_inv)

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
                    series_id=window.series_id, is_anomaly=False, score=0.0, confidence=0.0,
                    method_votes={"cold_start": 0.0}, explanation="Cold start: insufficient data",
                    context={"reason": "auto_train_skipped", "n": window.size},
                )

        if window.is_empty or window.last_value is None:
            return AnomalyResult.normal(series_id=str(window.sensor_id))

        vote_kwargs = build_vote_context(window, self._temporal_stats)
        vote_kwargs["_managed_observe"] = True

        value = window.last_value
        if self._scaler is not None:
            try:
                value = float(np.asarray(self._scaler.transform([[value]])).flatten()[0])
                if "nd_features" in vote_kwargs:
                    vote_kwargs["nd_features"][0, 0] = value
            except Exception as exc:
                logger.warning("scaler_transform_failed", extra={"error": str(exc)})
        votes: dict[str, float] = {}
        effective_weights = self._strategy._get_effective_weights()
        for detector in self._sub_detectors:
            if not detector.is_trained:
                continue
            # Optimización de latencia: si el detector es LOF y su peso efectivo es <= 0, no ejecutar sklearn k-NN
            if detector.method_name == "local_outlier_factor" and effective_weights.get("local_outlier_factor", 0.0) <= 0.0:
                votes[detector.method_name] = 0.0
                continue
            try:
                if self._calibration_layer is not None:
                    raw_s = detector.raw_score(value, **vote_kwargs)
                    v = self._calibration_layer.transform(detector.method_name, raw_s)
                else:
                    v = detector.vote(value, **vote_kwargs)
                if v is not None:
                    votes[detector.method_name] = v
            except Exception as exc:
                logger.debug(
                    "sub_detector_vote_failed",
                    extra={"detector": detector.method_name, "error": str(exc)},
                )

        final_score = self._strategy.combine(votes)
        is_anomaly = self._strategy.is_anomaly(final_score, votes=votes)
        if self._calibration_layer is not None:
            self._calibration_layer.observe(value, is_anomaly_active=is_anomaly)
        for detector in self._sub_detectors:
            if hasattr(detector, "observe"):
                try:
                    detector.observe(value, is_anomaly_active=is_anomaly)
                except Exception as exc:
                    logger.debug(
                        "sub_detector_observe_failed",
                        extra={"detector": detector.method_name, "error": str(exc)},
                    )
        confidence = self._strategy.confidence(votes)
        severity = ThresholdPolicy.default().classify_score(final_score)

        z = compute_z_score(value, self._stats.mean, self._stats.std)
        vel_z = extract_vel_z(window, self._temporal_stats)
        acc_z = extract_acc_z(window, self._temporal_stats)
        explanation = build_anomaly_explanation(
            votes, z_score=z, vel_z_score=vel_z, acc_z_score=acc_z,
        )

        return AnomalyResult(
            series_id=str(window.sensor_id), is_anomaly=is_anomaly, score=final_score,
            method_votes=votes, confidence=confidence, explanation=explanation, severity=severity,
        )

    def is_trained(self) -> bool:
        return self._trained_flag
