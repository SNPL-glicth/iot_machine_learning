"""Accuracy mixin for BayesianWeightTracker."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

_ERROR_STORE_MIN_SAMPLES: int = 30
_PERCENTILE_SAMPLE_SIZE: int = 100


class AccuracyMixin:
    """Mixin providing accuracy computation functionality."""

    # Declare expected parent attributes for mypy
    _error_store: Any
    _error_history: Dict[str, List[float]]

    def compute_accuracy(
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

    def _compute_robust_cap(self, errors: List[float]) -> float:
        """Compute robust error cap using percentile."""
        if not errors:
            return float("inf")
        sorted_errors = sorted(errors)
        p90_idx = int(len(sorted_errors) * 0.9)
        return sorted_errors[min(p90_idx, len(sorted_errors) - 1)]
