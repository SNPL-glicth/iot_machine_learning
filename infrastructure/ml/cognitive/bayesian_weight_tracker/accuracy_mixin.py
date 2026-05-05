"""Accuracy computation mixin for BayesianWeightTracker."""
from __future__ import annotations
import logging
import math
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

_ERROR_STORE_PERCENTILE: float = 99.0
_ERROR_STORE_CAP_MULTIPLIER: float = 3.0
_ERROR_STORE_MIN_SAMPLES: int = 10
_PERCENTILE_SAMPLE_SIZE: int = 100
_MAD_K_SMALL: float = 9.0
_MAD_K_LARGE: float = 6.0
_MAD_EPSILON: float = 1e-12


class AccuracyMixin:
    """Mixin providing accuracy computation with robust MAD-based capping."""

    def _compute_robust_cap(self, history: List[float]) -> float:
        """Robust outlier cap using Median Absolute Deviation (MAD)."""
        if not history:
            return float("inf")
        arr = np.asarray(history, dtype=np.float64)
        n = len(arr)
        median = float(np.median(arr))
        abs_dev = np.abs(arr - median)
        mad = float(np.median(abs_dev))
        k = _MAD_K_SMALL if n < 50 else _MAD_K_LARGE
        if mad < _MAD_EPSILON:
            mean_ad = float(np.mean(abs_dev))
            if mean_ad < _MAD_EPSILON:
                return median * 2.0 if median > 0 else float("inf")
            return median + k * mean_ad
        return median + k * mad

    def _compute_accuracy(
        self,
        prediction_error: float,
        regime: str,
        series_id: Optional[str] = None,
        engine_name: Optional[str] = None,
    ) -> float:
        abs_error = float(abs(prediction_error))
        if not math.isfinite(abs_error):
            abs_error = float("inf")
        if (
            self._error_store is not None
            and series_id is not None
            and engine_name is not None
        ):
            recent = self._error_store.get_recent(
                series_id, engine_name, _PERCENTILE_SAMPLE_SIZE
            )
            if len(recent) >= _ERROR_STORE_MIN_SAMPLES:
                cap = self._compute_robust_cap(recent)
                if 0.0 < cap < float("inf"):
                    abs_error = min(abs_error, cap)
            return 1.0 / (1.0 + abs_error)
        history = self._error_history.get(regime, [])
        if len(history) >= _ERROR_STORE_MIN_SAMPLES:
            cap = self._compute_robust_cap(history)
            if 0.0 < cap < float("inf"):
                abs_error = min(abs_error, cap)
        if regime not in self._error_history:
            self._error_history[regime] = []
        self._error_history[regime].append(abs_error)
        if len(self._error_history[regime]) > _PERCENTILE_SAMPLE_SIZE:
            self._error_history[regime].pop(0)
        accuracy = 1.0 / (1.0 + abs_error)
        return float(max(0.0, min(1.0, accuracy)))
