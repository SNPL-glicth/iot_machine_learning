"""IsolationForest sub-detector — detección por aislamiento global.

Una responsabilidad: evaluar si un valor es un outlier global
usando IsolationForest de sklearn.

Dependencia opcional: sklearn. Si no está disponible, vote() retorna None.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from ..core.protocol import SubDetector

logger = logging.getLogger(__name__)


class IsolationForestDetector(SubDetector):
    """Sub-detector basado en IsolationForest (sklearn).

    Attributes:
        _contamination: Fracción esperada de anomalías.
        _n_estimators: Número de árboles.
        _random_state: Semilla para reproducibilidad.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self._contamination = contamination
        self._n_estimators = n_estimators
        self._random_state = random_state
        self._model: object = None

    @property
    def method_name(self) -> str:
        return "isolation_forest"

    def train(self, values: List[float], **kwargs: object) -> None:
        try:
            from sklearn.ensemble import IsolationForest
            import numpy as np

            X = np.array(values).reshape(-1, 1)
            model = IsolationForest(
                contamination=self._contamination,
                random_state=self._random_state,
                n_estimators=self._n_estimators,
            )
            model.fit(X)
            self._model = model
            logger.debug(
                "if_detector_trained",
                extra={"n_points": len(values), "dims": 1},
            )
        except ImportError:
            logger.warning("sklearn_not_available_if_disabled")
            self._model = None

    def vote(self, value: float, **kwargs: object) -> Optional[float]:
        if self._model is None:
            return None
        try:
            import numpy as np
            X = np.array([[value]])
            score = self._model.decision_function(X)[0]
            return 1.0 if score < 0 else 0.0
        except Exception:
            return 0.0

    @property
    def is_trained(self) -> bool:
        return self._model is not None


class IsolationForestNDDetector(SubDetector):
    """Sub-detector IsolationForest N-dimensional (magnitud + temporal).

    Entrena sobre una feature matrix [value, velocity, acceleration].
    """

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        random_state: int = 42,
        min_training_points: int = 50,
    ) -> None:
        self._contamination = contamination
        self._n_estimators = n_estimators
        self._random_state = random_state
        self._min_training_points = min_training_points
        self._model: object = None

    @property
    def method_name(self) -> str:
        return "isolation_forest_temporal"

    def train(self, values: List[float], **kwargs: object) -> None:
        timestamps = kwargs.get("timestamps")
        if timestamps is None:
            return
        feature_matrix = self._build_features(values, list(timestamps))
        if feature_matrix is None:
            return
        try:
            from sklearn.ensemble import IsolationForest

            model = IsolationForest(
                contamination=self._contamination,
                random_state=self._random_state,
                n_estimators=self._n_estimators,
            )
            model.fit(feature_matrix)
            self._model = model
            logger.debug(
                "if_nd_detector_trained",
                extra={"n_points": len(values)},
            )
        except (ImportError, Exception) as exc:
            logger.warning(
                "if_temporal_training_failed", extra={"error": str(exc)}
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
            from iot_machine_learning.domain.validators.temporal_features import (
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
