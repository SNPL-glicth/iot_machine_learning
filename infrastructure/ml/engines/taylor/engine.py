"""Taylor prediction engine — thin orchestrator shell.

All logic lives in ``prediction_pipeline.py`` and ``engine_helpers.py``
to keep this file under 180 lines.

FASE 2 enhancements:
- Coefficient caching with TTL (fixes CRIT-1)
- MAE/RMSE tracking for confidence adjustment (fixes CRIT-2)
- Temporal gap detection (fixes CRIT-3)
"""

from __future__ import annotations

import logging
from typing import List, Optional

from iot_machine_learning.infrastructure.ml.interfaces import (
    PredictionEngine,
    PredictionResult,
)
from iot_machine_learning.infrastructure.ml.cognitive.hyperparameters import (
    HyperparameterAdaptor,
)

from .types import DerivativeMethod
from .coefficient_cache import TaylorCoefficientCache
from .performance_tracker import TaylorPerformanceTracker
from .gap_detector import TemporalGapDetector
from .prediction_pipeline import run_taylor_prediction

logger = logging.getLogger(__name__)

_TREND_THRESHOLD: float = 0.01
_MIN_CONFIDENCE: float = 0.3
_MAX_CONFIDENCE: float = 0.95
_CLAMP_MARGIN_PCT: float = 0.3


class TaylorPredictionEngine(PredictionEngine):
    """Taylor-series prediction engine.

    See ``taylor/`` package for mathematical specification.
    Configurable derivative method: backward, central, least_squares.

    FASE 2 features:
    - Coefficient caching (enable_cache=True)
    - Performance tracking for confidence (enable_tracking=True)
    - Gap detection (enable_gap_detection=True)
    """

    def __init__(
        self,
        order: int = 2,
        horizon: int = 1,
        *,
        derivative_method: DerivativeMethod = DerivativeMethod.BACKWARD,
        trend_threshold: float = _TREND_THRESHOLD,
        min_confidence: float = _MIN_CONFIDENCE,
        max_confidence: float = _MAX_CONFIDENCE,
        clamp_margin_pct: float = _CLAMP_MARGIN_PCT,
        enable_cache: bool = True,
        cache_ttl_seconds: int = 300,
        enable_tracking: bool = True,
        enable_gap_detection: bool = True,
        series_id: Optional[str] = None,
        hyperparameter_adaptor: Optional[HyperparameterAdaptor] = None,
        physical_min: Optional[float] = None,
        physical_max: Optional[float] = None,
    ) -> None:
        if horizon < 1:
            raise ValueError(f"horizon debe ser >= 1, recibido {horizon}")
        self._order = max(1, min(order, 3))
        self._horizon = horizon
        self._method = derivative_method
        self._trend_threshold = trend_threshold
        self._min_confidence = min_confidence
        self._max_confidence = max_confidence
        self._clamp_margin_pct = clamp_margin_pct
        self._series_id = series_id
        self._physical_min = physical_min
        self._physical_max = physical_max
        self._hyperparams = hyperparameter_adaptor

        self._cache = (
            TaylorCoefficientCache(ttl_seconds=cache_ttl_seconds)
            if enable_cache
            else None
        )
        self._tracker = (
            TaylorPerformanceTracker() if enable_tracking else None
        )
        self._gap_detector = (
            TemporalGapDetector() if enable_gap_detection else None
        )

        if order != self._order:
            logger.warning(
                "taylor_order_clamped",
                extra={"requested": order, "effective": self._order},
            )

    @property
    def name(self) -> str:
        return "taylor_finite_differences"

    def can_handle(self, n_points: int) -> bool:
        return n_points >= self._order + 2

    def predict(
        self,
        values: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> PredictionResult:
        return run_taylor_prediction(self, values, timestamps)

    def supports_uncertainty(self) -> bool:
        return False

    def record_actual(self, predicted: float, actual: float) -> None:
        """Record actual value for performance tracking (FASE 2)."""
        if self._tracker:
            self._tracker.record_error(predicted, actual)
            logger.debug(
                "taylor_actual_recorded",
                extra={
                    "predicted": round(predicted, 4),
                    "actual": round(actual, 4),
                    "error": round(abs(predicted - actual), 4),
                },
            )
