from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from iot_machine_learning.ml_service.config.ml_config import AnomalyConfig
from iot_machine_learning.ml_service.models.anomaly_model import AnomalyModel

logger = logging.getLogger(__name__)


class IsolationForestTrainer:
    """Entrena y aplica Isolation Forest por sensor sobre residuales.

    Supports model serialization via joblib to avoid retraining on each cycle.
    """

    def __init__(
        self,
        cfg: AnomalyConfig,
        model_storage_path: Optional[str] = None,
        enable_serialization: bool = True,
    ) -> None:
        self.cfg = cfg
        self._estimators: dict[int, IsolationForest] = {}
        self._models: dict[int, AnomalyModel] = {}
        self._enable_serialization = enable_serialization
        self._storage_path = Path(model_storage_path) if model_storage_path else None
        
        # Create storage directory if needed
        if self._storage_path and not self._storage_path.exists():
            self._storage_path.mkdir(parents=True, exist_ok=True)
            logger.info("[ISO_TRAINER] Created model storage at %s", self._storage_path)

    def _get_model_path(self, sensor_id: int) -> Optional[Path]:
        """Get the file path for a sensor's serialized model."""
        if self._storage_path is None:
            return None
        return self._storage_path / f"iso_forest_sensor_{sensor_id}.joblib"

    def _save_model(self, sensor_id: int, iso: IsolationForest, model: AnomalyModel) -> bool:
        """Serialize model to disk using joblib."""
        if not self._enable_serialization:
            return False
        
        path = self._get_model_path(sensor_id)
        if path is None:
            return False
        
        try:
            joblib.dump({
                "estimator": iso,
                "model": model,
                "config": {
                    "contamination": self.cfg.contamination,
                    "n_estimators": self.cfg.n_estimators,
                    "random_state": self.cfg.random_state,
                }
            }, path)
            logger.debug("[ISO_TRAINER] Saved model for sensor=%s to %s", sensor_id, path)
            return True
        except Exception as e:
            logger.warning("[ISO_TRAINER] Failed to save model for sensor=%s: %s", sensor_id, e)
            return False

    def _load_model(self, sensor_id: int) -> tuple[IsolationForest, AnomalyModel] | None:
        """Deserialize model from disk using joblib."""
        if not self._enable_serialization:
            return None
        
        path = self._get_model_path(sensor_id)
        if path is None or not path.exists():
            return None
        
        try:
            data = joblib.load(path)
            
            # Validate config compatibility
            saved_config = data.get("config", {})
            if (
                saved_config.get("contamination") != self.cfg.contamination or
                saved_config.get("n_estimators") != self.cfg.n_estimators or
                saved_config.get("random_state") != self.cfg.random_state
            ):
                logger.debug(
                    "[ISO_TRAINER] Config mismatch for sensor=%s, retraining needed", 
                    sensor_id
                )
                return None
            
            logger.debug("[ISO_TRAINER] Loaded model for sensor=%s from %s", sensor_id, path)
            return data["estimator"], data["model"]
        except Exception as e:
            logger.warning("[ISO_TRAINER] Failed to load model for sensor=%s: %s", sensor_id, e)
            return None

    def fit_for_sensor(self, sensor_id: int, residuals: np.ndarray) -> AnomalyModel | None:
        """Train or load IsolationForest model for a sensor.
        
        First attempts to load existing serialized model. If not available
        or incompatible, trains a new model and serializes it.
        """
        if residuals.size < 10:
            return None
        
        # Try to load existing model first
        loaded = self._load_model(sensor_id)
        if loaded is not None:
            iso, model = loaded
            self._estimators[sensor_id] = iso
            self._models[sensor_id] = model
            logger.debug("[ISO_TRAINER] Using cached model for sensor=%s", sensor_id)
            return model

        X = residuals.reshape(-1, 1)
        iso = IsolationForest(
            contamination=self.cfg.contamination,
            n_estimators=self.cfg.n_estimators,
            random_state=self.cfg.random_state,
        )
        iso.fit(X)

        scores = iso.score_samples(X)
        threshold = float(np.percentile(scores, 100 * self.cfg.contamination))
        score_min = float(scores.min())
        score_max = float(scores.max())

        model = AnomalyModel(
            sensor_id=sensor_id,
            threshold_score=threshold,
            score_min=score_min,
            score_max=score_max,
        )
        self._estimators[sensor_id] = iso
        self._models[sensor_id] = model
        
        # Serialize the trained model
        self._save_model(sensor_id, iso, model)
        
        return model

    def score_new_point(self, sensor_id: int, residual: float) -> tuple[float, bool]:
        iso = self._estimators.get(sensor_id)
        model = self._models.get(sensor_id)
        if iso is None or model is None:
            # Si no hay modelo entrenado, no marcamos anomalía.
            return 0.0, False

        X = np.array([[residual]], dtype=float)
        raw_score = float(iso.score_samples(X)[0])

        # Normalización a [0,1]: 0 = comportamiento normal, 1 = altamente anómalo.
        # IsolationForest da scores más negativos cuanto más anómalo es el punto.
        if raw_score >= model.threshold_score:
            norm_score = 0.0
        else:
            denom = max(model.threshold_score - model.score_min, 1e-6)
            norm_score = (model.threshold_score - raw_score) / denom
            if norm_score > 1.0:
                norm_score = 1.0

        is_anom = raw_score < model.threshold_score
        return norm_score, is_anom
