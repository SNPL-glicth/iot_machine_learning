# EXPERIMENTAL — NO USAR EN PRODUCCIÓN
# Limitaciones conocidas:
# - Requiere mínimo 500 puntos para ser estadísticamente válido
# - Con 80-100 puntos tiene sobreajuste extremo (~100 params/muestra)
# - Sin clamping físico: puede predecir valores fuera de rango
# - Confidence heurístico: no refleja incertidumbre real
# - Retrain batch de 50-150ms cada 50 preds: prohibitivo en IoT masivo
# - Sin detección de cambio de régimen
# Ver: docs/ENGINES.md sección "Experimental engines"
# Reimplementar con: min 500pts, Fourier features, TimeSeriesSplit,
# early stopping, clamping, confidence real vía holdout MAE

"""LightGBM prediction engine (P5).

Gradient-boosting regressor for non-linear patterns in IoT time series.
Optional dependency: lightgbm is imported lazily with graceful fallback.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from iot_machine_learning.infrastructure.ml.engines.core.factory import (
    PredictionEngine,
    PredictionResult,
    register_engine,
)

from .feature_builder import build_feature_vector, build_training_matrix

logger = logging.getLogger(__name__)

# Lazy optional import
try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None  # type: ignore[misc]


@register_engine("lightgbm")
class LightGBMPredictionEngine(PredictionEngine):
    """LightGBM-based prediction engine for non-linear IoT signals.

    P5 constraints:
    - Min 80 points to train (``min_train_points``).
    - When lightgbm is unavailable, ``can_handle()`` returns False and
      ``predict()`` returns confidence=0.0 explicitly.
    - Feature builder is stateless; model state lives in this engine.
    - Online learning: ``record_actual()`` accumulates data; retrain every
      ``retrain_every`` calls.
    """

    def __init__(
        self,
        min_train_points: int = 80,
        retrain_every: int = 50,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        n_estimators: int = 50,
        series_id: Optional[str] = None,
    ) -> None:
        if min_train_points < 20:
            raise ValueError(f"min_train_points must be >= 20, got {min_train_points}")
        if retrain_every < 10:
            raise ValueError(f"retrain_every must be >= 10, got {retrain_every}")
        self._min_train_points = min_train_points
        self._retrain_every = retrain_every
        self._learning_rate = learning_rate
        self._num_leaves = num_leaves
        self._n_estimators = n_estimators
        self._series_id = series_id

        self._model: Any = None
        self._feature_names: List[str] = []
        self._history_values: List[float] = []
        self._history_timestamps: List[float] = []
        self._record_count = 0

    @property
    def name(self) -> str:
        return "lightgbm_regressor"

    def can_handle(self, n_points: int) -> bool:
        # Need min_train_points + 1 to build at least one supervised pair
        if lgb is None:
            return False
        return n_points >= self._min_train_points + 1

    def predict(
        self,
        values: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> PredictionResult:
        """Predict using LightGBM regressor or graceful fallback.

        - Requires ``min_train_points + 1`` values to build supervised pairs.
        - When ``lightgbm`` is not installed, returns confidence=0.0.

        Args:
            values: Time-series window.
            timestamps: Optional monotonic timestamps.

        Returns:
            PredictionResult with fallback metadata when insufficient data.

        Raises:
            ValueError: If values is empty.
        """
        # Fallback when lightgbm is unavailable
        if lgb is None:
            return PredictionResult(
                predicted_value=values[-1] if values else 0.0,
                confidence=0.0,
                trend="stable",
                metadata={
                    "fallback": "lightgbm_not_installed",
                    "lightgbm_available": False,
                },
            )

        if not values:
            raise ValueError("values cannot be empty")

        n = len(values)
        if not self.can_handle(n):
            # Cold-start fallback: use last value
            return PredictionResult(
                predicted_value=values[-1] if values else 0.0,
                confidence=0.3,
                trend="stable",
                metadata={
                    "fallback": "insufficient_data",
                    "needed": self._min_train_points + 1,
                    "provided": n,
                },
            )

        # Retrain if model is stale or missing
        if self._model is None or self._should_retrain():
            self._train(values, timestamps)

        # If training failed or still no model, fallback
        if self._model is None:
            return PredictionResult(
                predicted_value=values[-1] if values else 0.0,
                confidence=0.3,
                trend="stable",
                metadata={"fallback": "training_failed"},
            )

        # Inference
        feats = build_feature_vector(values, timestamps)
        row = [feats.get(k, 0.0) for k in self._feature_names]
        import numpy as np

        X = np.array([row])
        predicted = float(self._model.predict(X)[0])

        # Trend from features
        trend = "stable"
        if feats.get("delta_1", 0.0) > 0.01:
            trend = "up"
        elif feats.get("delta_1", 0.0) < -0.01:
            trend = "down"

        # Confidence heuristic: higher when training data was abundant
        confidence = min(0.9, 0.5 + (n / (self._min_train_points * 3)))

        return PredictionResult(
            predicted_value=predicted,
            confidence=confidence,
            trend=trend,  # type: ignore[arg-type]
            metadata={
                "fallback": None,
                "model_trained": self._model is not None,
                "feature_count": len(self._feature_names),
                "training_size": n,
            },
        )

    def supports_uncertainty(self) -> bool:
        return False

    def record_actual(self, predicted: float, actual: float) -> None:
        """Accumulate actual values for periodic online retraining."""
        if not (math.isfinite(predicted) and math.isfinite(actual)):
            logger.warning(
                "lightgbm_record_actual_skipped",
                extra={"predicted": predicted, "actual": actual},
            )
            return
        try:
            self._history_values.append(actual)
            self._record_count += 1
        except Exception as exc:
            logger.warning("lightgbm_record_actual_failed", extra={"error": str(exc)})

        # Trigger retrain if threshold reached
        if self._record_count >= self._retrain_every:
            self._record_count = 0
            # Mark model as stale so next predict() retrains
            self._model = None

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _should_retrain(self) -> bool:
        return self._model is None

    def _train(
        self,
        values: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> None:
        """Train LightGBM regressor on the supplied window."""
        if lgb is None:
            return

        # Combine history with current window for richer training
        all_values = self._history_values + values
        all_ts = (
            self._history_timestamps + timestamps
            if timestamps is not None
            else None
        )

        X, feature_names, y = build_training_matrix(
            all_values, all_ts, min_points=self._min_train_points
        )
        if not X or not feature_names:
            logger.warning("lightgbm_training_skipped: insufficient data")
            return

        try:
            import numpy as np

            self._feature_names = feature_names
            X_np = np.array(X)
            y_np = np.array(y)

            self._model = lgb.LGBMRegressor(
                learning_rate=self._learning_rate,
                num_leaves=self._num_leaves,
                n_estimators=self._n_estimators,
                verbosity=-1,
            )
            self._model.fit(X_np, y_np)

            logger.info(
                "lightgbm_trained",
                extra={
                    "series_id": self._series_id,
                    "samples": len(y),
                    "features": len(feature_names),
                },
            )
        except Exception as exc:
            logger.warning("lightgbm_training_failed", extra={"error": str(exc)})
            self._model = None
