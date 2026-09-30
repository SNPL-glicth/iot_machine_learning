"""IsolationForest sub-detector — detección por aislamiento global.

Una responsabilidad: evaluar si un valor es un outlier global
usando IsolationForest de sklearn.

Dependencia opcional: sklearn. Si no está disponible, vote() retorna None.

MATH-SEV-1: Contamination adaptativa basada en tasa histórica de anomalías.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from core.drift.adaptive_contamination import AdaptiveContamination, ContaminationHysteresisConfig
from core.parameters.numerical_constants import EPSILON, STAT_THRESHOLDS

from ..core.protocol import SubDetector

logger = logging.getLogger(__name__)

# MATH-SEV-1: Bounds para contamination adaptativa (use STAT_THRESHOLDS)
_MIN_SAMPLES_FOR_ADAPTIVE = 100


class IsolationForestDetector(SubDetector):
    """Sub-detector basado en IsolationForest (sklearn).

    Attributes:
        _contamination: Fracción esperada de anomalías (base).
        _n_estimators: Número de árboles.
        _random_state: Semilla para reproducibilidad.
        _adaptive: Si True, estima contamination de datos históricos.

    MATH-SEV-1: Contamination adaptativa reduce falsos positivos/negativos.
    """

    def __init__(
        self,
        contamination: float | None = None,
        n_estimators: int = 100,
        random_state: int = 42,
        adaptive: bool = True,  # MATH-SEV-1
        use_adaptive_contamination: bool = False,  # NUEVO: Fase 4
    ) -> None:
        # Use STAT_THRESHOLDS default if not provided
        if contamination is None:
            contamination = STAT_THRESHOLDS.CONTAMINATION_DEFAULT

        self._contamination = contamination
        self._n_estimators = n_estimators
        self._random_state = random_state
        self._adaptive = adaptive  # MATH-SEV-1
        self._model: Any = None

        # NUEVO: Adaptive contamination con hysteresis
        self._use_adaptive_contamination = use_adaptive_contamination
        if use_adaptive_contamination:
            self.adaptive_contamination = AdaptiveContamination(
                hysteresis_config=ContaminationHysteresisConfig(
                    min_samples=_MIN_SAMPLES_FOR_ADAPTIVE,
                ),
            )
        else:
            self.adaptive_contamination = None

    @property
    def method_name(self) -> str:
        return "isolation_forest"

    def train(self, values: list[float], **kwargs: object) -> None:
        try:
            import numpy as np
            from sklearn.ensemble import IsolationForest

            arr = np.array(values, dtype=float)
            valid_arr = arr[np.isfinite(arr)]
            if len(valid_arr) < 5:
                logger.warning("if_insufficient_valid_points")
                self._model = None
                return

            X = valid_arr.reshape(-1, 1)

            # MATH-SEV-1: Estimate contamination adaptively
            effective_contamination = self._contamination
            if self._adaptive and len(valid_arr) >= _MIN_SAMPLES_FOR_ADAPTIVE:
                estimated = self._estimate_contamination(valid_arr.tolist())
                if estimated is not None:
                    effective_contamination = estimated
                    logger.info(
                        "if_detector_adaptive_contamination",
                        extra={
                            "base": self._contamination,
                            "estimated": estimated,
                            "n_samples": len(valid_arr),
                        }
                    )

            model = IsolationForest(
                contamination=float(effective_contamination),  # type: ignore[arg-type]
                random_state=self._random_state,
                n_estimators=self._n_estimators,
            )
            model.fit(X)
            self._model = model
            # Guardar scores de training para calibración continua
            self._training_scores = self._model.decision_function(X).flatten()
            logger.debug(
                "if_detector_trained",
                extra={
                    "n_points": len(valid_arr),
                    "dims": 1,
                    "contamination": effective_contamination,
                },
            )
        except ImportError:
            logger.warning("sklearn_not_available_if_disabled")
            self._model = None

    def vote(self, value: float, **kwargs: object) -> float | None:
        if self._model is None:
            return None

        # Get regime for contextual scoring (ETAPA 2)
        regime = kwargs.get("regime")
        regime_config = kwargs.get("regime_config")

        try:
            import numpy as np
            if np.isnan(value) or np.isinf(value):
                return 0.0
            X = np.array([[value]])
            score = float(self._model.decision_function(X)[0])
            if np.isnan(score) or score >= 0.0:
                return 0.0

            # Mapeo suave y continuo de evidencia de aislamiento:
            # Inliers tienen score >= 0. Puntos con score < 0 escalan continuamente
            # hasta saturar a 1.0 en lejanía clara (score <= -0.10).
            vote_continuous = min(1.0, -score / 0.10)

            multiplier = 1.0
            if regime and regime_config:
                if regime == "VOLATILE_PEAK":
                    multiplier = getattr(regime_config, "peak_load_multiplier", 1.0)
                elif regime == "STARTUP" or regime == "SHUTDOWN":
                    multiplier = getattr(regime_config, "transition_multiplier", 1.0)
                elif regime == "ANOMALOUS_REGIME":
                    multiplier = 1.0
            elif regime:
                if regime == "VOLATILE_PEAK":
                    multiplier = 0.7
                elif regime == "STARTUP" or regime == "SHUTDOWN":
                    multiplier = 0.8

            return float(np.clip(vote_continuous * multiplier, 0.0, 1.0))
        except Exception:
            return 0.0

    def raw_score(self, value: float, **kwargs: object) -> float | None:
        if self._model is None:
            return None
        try:
            import numpy as np
            if np.isnan(value) or np.isinf(value):
                return None
            return float(self._model.decision_function(np.array([[value]]))[0])
        except Exception:
            return None

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def update_contamination(self, values: list[float]) -> float | None:
        """Actualiza contamination y re-entrena si es necesario.

        Args:
            values: Valores recientes para re-entrenamiento si es necesario.

        Returns:
            Nuevo valor de contamination, o None si no se actualizó.
        """
        if not self.adaptive_contamination:
            return None

        # Actualizar contamination
        new_contamination = self.adaptive_contamination.update_contamination()

        # Re-entrenar si el cambio es significativo
        if self.adaptive_contamination.should_refit():
            logger.info(
                "if_detector_refitting",
                extra={
                    "old_contamination": self._contamination,
                    "new_contamination": new_contamination,
                    "change_percent": abs(new_contamination - self._contamination) / self._contamination * 100 if self._contamination > 0 else 0,
                },
            )
            self._contamination = new_contamination
            self.train(values)

        return new_contamination

    def _estimate_contamination(self, values: list[float]) -> float | None:
        """Estimate contamination rate from historical data (MATH-SEV-1).

        Uses z-scores to identify anomalies: |z| > Z_SCORE_LOWER is considered anomalous.

        Args:
            values: Historical values.

        Returns:
            Estimated contamination rate clamped to [CONTAMINATION_MIN, CONTAMINATION_MAX], or None if fails.

        Applies OCP: Subclasses can override this method for custom estimation.
        """
        try:
            import numpy as np

            if len(values) < _MIN_SAMPLES_FOR_ADAPTIVE:
                return None

            arr = np.array(values)

            # Remove NaN/Inf
            arr = arr[np.isfinite(arr)]
            if len(arr) < _MIN_SAMPLES_FOR_ADAPTIVE:
                return None

            # Calculate z-scores
            mean = np.mean(arr)
            std = np.std(arr)

            if std < EPSILON.DIVISION:  # Constant signal
                return STAT_THRESHOLDS.CONTAMINATION_MIN

            z_scores = np.abs((arr - mean) / std)

            # Count anomalies (|z| > threshold)
            n_anomalies = np.sum(z_scores > STAT_THRESHOLDS.Z_SCORE_LOWER)
            contamination_rate = n_anomalies / len(arr)

            # Clamp to bounds
            clamped = max(
                STAT_THRESHOLDS.CONTAMINATION_MIN,
                min(STAT_THRESHOLDS.CONTAMINATION_MAX, contamination_rate)
            )

            return clamped

        except Exception as exc:
            logger.warning(
                "contamination_estimation_failed",
                extra={"error": str(exc)},
            )
            return None


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
        self._model: Any = None

    @property
    def method_name(self) -> str:
        return "isolation_forest_temporal"

    def train(self, values: list[float], **kwargs: object) -> None:
        timestamps = kwargs.get("timestamps")
        if not isinstance(timestamps, Iterable):
            return
        feature_matrix = self._build_features(values, list(timestamps))
        if feature_matrix is None:
            return
        try:
            from sklearn.ensemble import IsolationForest

            model = IsolationForest(
                contamination=float(self._contamination),  # type: ignore[arg-type]
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

    def vote(self, value: float, **kwargs: object) -> float | None:
        if self._model is None:
            return None

        # Try to use DynamicFeatures for multi-dimensional features (v2.0.0)
        dyn_raw = kwargs.get("dynamic_features")
        dynamic_features: dict[str, Any] | None = dyn_raw if isinstance(dyn_raw, dict) else None
        if dynamic_features is not None:
            try:
                import numpy as np
                # Build feature vector from DynamicFeatures
                feature_vector = []
                feature_vector.append(value)  # current value
                if dynamic_features.get("derivative") is not None:
                    feature_vector.append(dynamic_features["derivative"])
                if dynamic_features.get("second_derivative") is not None:
                    feature_vector.append(dynamic_features["second_derivative"])
                if dynamic_features.get("rolling_std_1h") is not None:
                    feature_vector.append(dynamic_features["rolling_std_1h"])

                if len(feature_vector) > 1:
                    X = np.array([feature_vector])
                    score = self._model.decision_function(X)[0]
                    return max(0.0, min(1.0, -score / 3.0))
            except Exception:
                pass  # Fallback to nd_features

        # Fallback to nd_features (v1.0.0)
        features = kwargs.get("nd_features")
        if features is None:
            return None
        try:
            score = self._model.decision_function(features)[0]
            return max(0.0, min(1.0, -score / 3.0))
        except Exception:
            return 0.0

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def _build_features(
        self, values: list[float], timestamps: list[float]
    ) -> Any:
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
