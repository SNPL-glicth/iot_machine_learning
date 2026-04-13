"""Persistent anomaly detector — auto-saves/loads trained models.

Wraps VotingAnomalyDetector and automatically persists/loads
IsolationForest and LOF models to/from zenin_ml.ml_models.

FASE 1 — CRÍTICO: Fix ANOM-1 (models lost on restart).
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from iot_machine_learning.domain.entities.anomaly import AnomalyResult
from iot_machine_learning.domain.entities.sensor_reading import SensorWindow
from iot_machine_learning.domain.ports.anomaly_detection_port import AnomalyDetectionPort

from .core.detector import VotingAnomalyDetector
from .core.config import AnomalyDetectorConfig
from iot_machine_learning.infrastructure.persistence.sql.zenin_ml.model_repository import ModelRepository

logger = logging.getLogger(__name__)


class PersistentAnomalyDetector(AnomalyDetectionPort):
    """Anomaly detector with automatic model persistence.

    Wraps VotingAnomalyDetector and persists/loads sklearn models
    (IsolationForest, LOF) to/from zenin_ml.ml_models.

    Attributes:
        _detector: Inner VotingAnomalyDetector
        _model_repo: Repository for model persistence
        _series_id: Series identifier
        _domain_type: Domain type ('sensor', 'text', etc.)
        _auto_save: If True, auto-saves after training
        _auto_load: If True, auto-loads on first detect
    """

    def __init__(
        self,
        series_id: str,
        domain_type: str = "sensor",
        config: Optional[AnomalyDetectorConfig] = None,
        model_repo: Optional[ModelRepository] = None,
        auto_save: bool = True,
        auto_load: bool = True,
    ) -> None:
        """Initialize persistent detector.

        Args:
            series_id: Series identifier
            domain_type: Domain type
            config: Anomaly detector config
            model_repo: Model repository (creates default if None)
            auto_save: Auto-save models after training
            auto_load: Auto-load models on first detect
        """
        self._detector = VotingAnomalyDetector(config=config)
        self._model_repo = model_repo or ModelRepository()
        self._series_id = series_id
        self._domain_type = domain_type
        self._auto_save = auto_save
        self._auto_load = auto_load
        self._loaded = False

    @property
    def name(self) -> str:
        return "persistent_anomaly_detector"

    def train(
        self,
        historical_values: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> None:
        """Train detector and optionally persist models.

        Args:
            historical_values: Training data
            timestamps: Optional timestamps
        """
        start_time = time.time()

        # Train inner detector
        self._detector.train(historical_values, timestamps=timestamps)

        training_duration_ms = int((time.time() - start_time) * 1000)

        # Auto-save models if enabled
        if self._auto_save:
            self._save_models(
                training_points=len(historical_values),
                training_duration_ms=training_duration_ms,
            )

        self._loaded = True

    def detect(self, window: SensorWindow) -> AnomalyResult:
        """Detect anomalies, auto-loading models if needed.

        Args:
            window: Sensor window

        Returns:
            AnomalyResult
        """
        # Auto-load models on first detect if enabled
        if self._auto_load and not self._loaded:
            self._load_models()

        return self._detector.detect(window)

    def _save_models(
        self,
        training_points: int,
        training_duration_ms: int,
    ) -> None:
        """Save trained sklearn models to DB."""
        try:
            # Save each sub-detector that has a trained model
            for detector in self._detector._sub_detectors:
                if not detector.is_trained:
                    continue

                # Only save sklearn models (IF, LOF)
                if detector.method_name in [
                    "isolation_forest",
                    "isolation_forest_temporal",
                    "local_outlier_factor",
                    "lof_temporal",
                ]:
                    model_obj = getattr(detector, "_model", None)
                    if model_obj is not None:
                        # Extract hyperparameters
                        hyperparameters = self._extract_hyperparameters(detector)

                        self._model_repo.save_model(
                            model_name=detector.method_name,
                            series_id=self._series_id,
                            domain_type=self._domain_type,
                            model_obj=model_obj,
                            training_points=training_points,
                            hyperparameters=hyperparameters,
                            training_duration_ms=training_duration_ms,
                        )

            logger.info(
                "anomaly_models_persisted",
                extra={
                    "series_id": self._series_id,
                    "domain_type": self._domain_type,
                    "training_points": training_points,
                },
            )

        except Exception as exc:
            logger.error(
                "anomaly_models_save_failed",
                extra={
                    "series_id": self._series_id,
                    "error": str(exc),
                },
            )

    def _load_models(self) -> None:
        """Load trained sklearn models from DB."""
        try:
            loaded_count = 0

            for detector in self._detector._sub_detectors:
                if detector.method_name in [
                    "isolation_forest",
                    "isolation_forest_temporal",
                    "local_outlier_factor",
                    "lof_temporal",
                ]:
                    model_obj = self._model_repo.load_model(
                        series_id=self._series_id,
                        model_name=detector.method_name,
                    )

                    if model_obj is not None:
                        detector._model = model_obj
                        loaded_count += 1

            if loaded_count > 0:
                self._detector._trained_flag = True
                self._loaded = True

                logger.info(
                    "anomaly_models_loaded",
                    extra={
                        "series_id": self._series_id,
                        "models_loaded": loaded_count,
                    },
                )
            else:
                logger.debug(
                    "no_anomaly_models_found",
                    extra={"series_id": self._series_id},
                )

        except Exception as exc:
            logger.error(
                "anomaly_models_load_failed",
                extra={
                    "series_id": self._series_id,
                    "error": str(exc),
                },
            )

    def _extract_hyperparameters(self, detector) -> dict:
        """Extract hyperparameters from detector."""
        hyperparams = {}

        if hasattr(detector, "_contamination"):
            hyperparams["contamination"] = detector._contamination
        if hasattr(detector, "_n_estimators"):
            hyperparams["n_estimators"] = detector._n_estimators
        if hasattr(detector, "_random_state"):
            hyperparams["random_state"] = detector._random_state
        if hasattr(detector, "_max_neighbors"):
            hyperparams["max_neighbors"] = detector._max_neighbors
        if hasattr(detector, "_min_training_points"):
            hyperparams["min_training_points"] = detector._min_training_points

        return hyperparams
