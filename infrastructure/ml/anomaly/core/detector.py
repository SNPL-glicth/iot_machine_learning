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
        enable_asymmetric_dispatch: bool = False,
        meta_gate: Any | None = None,
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
        self._enable_asymmetric_dispatch = enable_asymmetric_dispatch
        self._meta_gate = meta_gate
        self._dispatcher: Any | None = None
        self._adapters: dict[str, Any] = {}

        if self._enable_asymmetric_dispatch:
            from iot_machine_learning.infrastructure.ml.moe.asymmetric import (
                AsymmetricDispatcher,
                SubDetectorExpertAdapter,
            )
            from iot_machine_learning.domain.entities.representation_evidence import (
                RepresentationLevel,
            )

            affinity_map = {
                "iqr": (RepresentationLevel.TEN_X, 0.01),
                "z_score": (RepresentationLevel.TEN_X, 0.02),
                "rolling_z": (RepresentationLevel.TWO_X, 0.05),
                "cumulative_residual": (RepresentationLevel.TWO_X, 0.05),
                "velocity_z": (RepresentationLevel.TWO_X, 0.02),
                "isolation_forest": (RepresentationLevel.RAW, 1.00),
                "isolation_forest_temporal": (RepresentationLevel.RAW, 1.00),
                "local_outlier_factor": (RepresentationLevel.RAW, 0.50),
                "acceleration_z": (RepresentationLevel.RAW, 0.02),
            }

            self._dispatcher = AsymmetricDispatcher()
            for d in self._sub_detectors:
                aff, cost = affinity_map.get(d.method_name, (RepresentationLevel.RAW, 1.0))
                adapter = SubDetectorExpertAdapter(
                    d, aff, calibration_layer=self._calibration_layer, scaler=self._scaler, compute_cost_estimate=cost
                )
                self._adapters[d.method_name] = adapter
                self._dispatcher.register_expert(adapter)

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

        if self._enable_asymmetric_dispatch and self._adapters:
            for ad in self._adapters.values():
                ad.scaler = self._scaler
                ad.calibration_layer = self._calibration_layer

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

        if self._enable_asymmetric_dispatch and self._dispatcher is not None:
            from iot_machine_learning.domain.entities.representation_evidence import (
                RepresentationLevel,
                SystemOperationalState,
            )

            for ad in self._adapters.values():
                ad.set_context(vote_kwargs)

            # 1. Despachar 10X (Resting/Invariantes) y 2X (Drift/Régimen)
            ev_10x = self._dispatcher.dispatch(RepresentationLevel.TEN_X, [window.last_value])
            ev_2x = self._dispatcher.dispatch(RepresentationLevel.TWO_X, [window.last_value])
            active_evidences = list(ev_10x + ev_2x)

            for ev in active_evidences:
                votes[ev.expert_name] = ev.anomaly_probability

            sum_fast = sum(
                votes.get(m, 0.0) * effective_weights.get(m, 0.0)
                for m in ["z_score", "cumulative_residual", "iqr", "velocity_z", "rolling_z"]
            )
            heavy_names = [
                m
                for m in ["isolation_forest", "isolation_forest_temporal", "local_outlier_factor", "acceleration_z"]
                if m in self._adapters
            ]
            max_remaining_heavy = sum(effective_weights.get(m, 0.0) for m in heavy_names)

            # 2. Enrutamiento condicional asimétrico
            if sum_fast + max_remaining_heavy < self._strategy._threshold:
                for m in heavy_names:
                    votes[m] = 0.0
            else:
                ev_raw = self._dispatcher.dispatch(RepresentationLevel.RAW, [window.last_value])
                active_evidences.extend(ev_raw)
                for ev in ev_raw:
                    votes[ev.expert_name] = ev.anomaly_probability

            final_score = self._strategy.combine(votes)
            is_anomaly = self._strategy.is_anomaly(final_score, votes=votes)

            # 3. Ville Gate si está conectado
            if self._meta_gate is not None:
                budget_ratio = 1.0 - (
                    self._dispatcher._total_cost_expended
                    / max(1.0, self._dispatcher._total_cost_hypothetical_full)
                )
                op_state = (
                    SystemOperationalState.RESTING
                    if sum_fast < 0.20
                    else SystemOperationalState.DRIFTING
                )
                gate_decision = self._meta_gate.evaluate_step(
                    step=window.size,
                    evidences=active_evidences,
                    operational_state=op_state,
                    budget_remaining_ratio=budget_ratio,
                )
                if "gate_decision" not in vote_kwargs:
                    vote_kwargs["gate_decision"] = gate_decision
        else:
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
