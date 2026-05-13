"""IsolationForest sub-detector — detección por aislamiento global.

Una responsabilidad: evaluar si un valor es un outlier global
usando IsolationForest de sklearn.

Dependencia opcional: sklearn. Si no está disponible, vote() retorna None.

MATH-SEV-1: Contamination adaptativa basada en tasa histórica de anomalías.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

from core.parameters.numerical_constants import EPSILON, STAT_THRESHOLDS
from core.drift.adaptive_contamination import AdaptiveContamination, ContaminationHysteresisConfig

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
        contamination: float = None,
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
        self._model: object = None
        
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

    def train(self, values: List[float], **kwargs: object) -> None:
        try:
            from sklearn.ensemble import IsolationForest
            import numpy as np

            X = np.array(values).reshape(-1, 1)
            
            # MATH-SEV-1: Estimate contamination adaptively
            effective_contamination = self._contamination
            if self._adaptive and len(values) >= _MIN_SAMPLES_FOR_ADAPTIVE:
                estimated = self._estimate_contamination(values)
                if estimated is not None:
                    effective_contamination = estimated
                    logger.info(
                        "if_detector_adaptive_contamination",
                        extra={
                            "base": self._contamination,
                            "estimated": estimated,
                            "n_samples": len(values),
                        }
                    )
            
            model = IsolationForest(
                contamination=effective_contamination,
                random_state=self._random_state,
                n_estimators=self._n_estimators,
            )
            model.fit(X)
            self._model = model
            logger.debug(
                "if_detector_trained",
                extra={
                    "n_points": len(values),
                    "dims": 1,
                    "contamination": effective_contamination,
                },
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
            is_anomaly = 1.0 if score < 0 else 0.0
            
            # NUEVO: Registrar detección para adaptive contamination
            if self.adaptive_contamination:
                self.adaptive_contamination.add_detection(is_anomaly == 1.0)
            
            return is_anomaly
        except Exception:
            return 0.0

    @property
    def is_trained(self) -> bool:
        return self._model is not None
    
    def update_contamination(self, values: List[float]) -> Optional[float]:
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
    
    def _estimate_contamination(self, values: List[float]) -> Optional[float]:
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
