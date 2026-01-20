from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from iot_machine_learning.ml_service.config.ml_config import AnomalyConfig
from iot_machine_learning.ml_service.models.anomaly_model import AnomalyModel


class IsolationForestTrainer:
    """Entrena y aplica Isolation Forest por sensor sobre residuales.

    Nota: el estimador de sklearn se mantiene en memoria mientras corre el batch.
    """

    def __init__(self, cfg: AnomalyConfig) -> None:
        self.cfg = cfg
        self._estimators: dict[int, IsolationForest] = {}
        self._models: dict[int, AnomalyModel] = {}

    def fit_for_sensor(self, sensor_id: int, residuals: np.ndarray) -> AnomalyModel | None:
        if residuals.size < 10:
            return None

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
