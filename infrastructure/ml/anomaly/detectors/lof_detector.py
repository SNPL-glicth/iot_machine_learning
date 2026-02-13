"""LocalOutlierFactor sub-detector — detección por densidad local.

Una responsabilidad: evaluar si un valor es un outlier local
usando LOF de sklearn.

Dependencia opcional: sklearn. Si no está disponible, vote() retorna None.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from ..detector_protocol import SubDetector

logger = logging.getLogger(__name__)


class LOFDetector(SubDetector):
    """Sub-detector basado en LocalOutlierFactor (sklearn).

    Attributes:
        _contamination: Fracción esperada de anomalías.
        _max_neighbors: Máximo de vecinos para LOF.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        max_neighbors: int = 20,
    ) -> None:
        self._contamination = contamination
        self._max_neighbors = max_neighbors
        self._model: object = None

    @property
    def method_name(self) -> str:
        return "local_outlier_factor"

    def train(self, values: List[float], **kwargs: object) -> None:
        try:
            from sklearn.neighbors import LocalOutlierFactor
            import numpy as np

            n = len(values)
            X = np.array(values).reshape(-1, 1)
            model = LocalOutlierFactor(
                n_neighbors=min(self._max_neighbors, n // 3),
                contamination=self._contamination,
                novelty=True,
            )
            model.fit(X)
            self._model = model
            logger.debug(
                "lof_detector_trained",
                extra={"n_points": n, "dims": 1},
            )
        except (ImportError, Exception) as exc:
            logger.warning(
                "lof_training_failed", extra={"error": str(exc)}
            )
            self._model = None

    def vote(self, value: float, **kwargs: object) -> Optional[float]:
        if self._model is None:
            return None
        try:
            import numpy as np
            X = np.array([[value]])
            score = self._model.decision_function(X)[0]
            return max(0.0, min(1.0, (-score - 1.0) / 2.0))
        except Exception:
            return 0.0

    @property
    def is_trained(self) -> bool:
        return self._model is not None


class LOFNDDetector(SubDetector):
    """Sub-detector LOF N-dimensional (magnitud + temporal).

    Entrena sobre una feature matrix [value, velocity, acceleration].
    """

    def __init__(
        self,
        contamination: float = 0.1,
        max_neighbors: int = 20,
        min_training_points: int = 50,
    ) -> None:
        self._contamination = contamination
        self._max_neighbors = max_neighbors
        self._min_training_points = min_training_points
        self._model: object = None

    @property
    def method_name(self) -> str:
        return "lof_temporal"

    def train(self, values: List[float], **kwargs: object) -> None:
        timestamps = kwargs.get("timestamps")
        if timestamps is None:
            return
        feature_matrix = self._build_features(values, list(timestamps))
        if feature_matrix is None:
            return
        try:
            from sklearn.neighbors import LocalOutlierFactor

            n_rows = feature_matrix.shape[0] if hasattr(feature_matrix, "shape") else len(values)
            model = LocalOutlierFactor(
                n_neighbors=min(self._max_neighbors, max(2, n_rows // 3)),
                contamination=self._contamination,
                novelty=True,
            )
            model.fit(feature_matrix)
            self._model = model
            logger.debug(
                "lof_nd_detector_trained",
                extra={"n_points": len(values)},
            )
        except (ImportError, Exception) as exc:
            logger.warning(
                "lof_temporal_training_failed", extra={"error": str(exc)}
            )

    def vote(self, value: float, **kwargs: object) -> Optional[float]:
        if self._model is None:
            return None
        features = kwargs.get("nd_features")
        if features is None:
            return None
        try:
            score = self._model.decision_function(features)[0]
            return 1.0 if score < 0 else 0.0
        except Exception:
            return 0.0

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def _build_features(
        self, values: List[float], timestamps: List[float]
    ) -> object:
        try:
            import numpy as np
            from .....domain.validators.temporal_features import (
                compute_temporal_features,
            )

            tf = compute_temporal_features(values, timestamps)
            if not tf.has_acceleration:
                return None

            n_acc = len(tf.accelerations)
            aligned_values = values[2 : 2 + n_acc]
            aligned_vels = tf.velocities[1 : 1 + n_acc]

            if len(aligned_values) != n_acc or len(aligned_vels) != n_acc:
                return None

            X = np.column_stack([
                np.array(aligned_values),
                np.array(aligned_vels),
                np.array(tf.accelerations),
            ])
            return X if X.shape[0] >= self._min_training_points else None
        except Exception:
            return None
