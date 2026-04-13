"""Taylor performance tracker — FASE 2.

Tracks prediction errors (MAE, RMSE) to incorporate into confidence calculation.

Fixes CRIT-2: Confianza artificial — now uses historical error.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for Taylor predictor."""

    mae: float
    rmse: float
    n_samples: int
    recent_mae: float
    recent_rmse: float


class TaylorPerformanceTracker:
    """Tracks Taylor prediction performance over time.

    Attributes:
        _errors: Deque of recent errors
        _max_history: Maximum errors to track
        _total_squared_error: Cumulative squared error
        _total_absolute_error: Cumulative absolute error
        _n_samples: Total samples tracked
    """

    def __init__(self, max_history: int = 100):
        """Initialize tracker.

        Args:
            max_history: Maximum errors to track
        """
        self._errors: Deque[float] = deque(maxlen=max_history)
        self._max_history = max_history
        self._total_squared_error = 0.0
        self._total_absolute_error = 0.0
        self._n_samples = 0

    def record_error(self, predicted: float, actual: float) -> None:
        """Record prediction error.

        Args:
            predicted: Predicted value
            actual: Actual value
        """
        error = abs(predicted - actual)
        squared_error = error ** 2

        self._errors.append(error)
        self._total_absolute_error += error
        self._total_squared_error += squared_error
        self._n_samples += 1

        logger.debug(
            "taylor_error_recorded",
            extra={
                "error": round(error, 4),
                "n_samples": self._n_samples,
            },
        )

    def get_metrics(self) -> Optional[PerformanceMetrics]:
        """Get performance metrics.

        Returns:
            PerformanceMetrics or None if no data
        """
        if self._n_samples == 0:
            return None

        # Overall metrics
        mae = self._total_absolute_error / self._n_samples
        rmse = math.sqrt(self._total_squared_error / self._n_samples)

        # Recent metrics (last N errors)
        if len(self._errors) > 0:
            recent_mae = sum(self._errors) / len(self._errors)
            recent_rmse = math.sqrt(sum(e ** 2 for e in self._errors) / len(self._errors))
        else:
            recent_mae = mae
            recent_rmse = rmse

        return PerformanceMetrics(
            mae=mae,
            rmse=rmse,
            n_samples=self._n_samples,
            recent_mae=recent_mae,
            recent_rmse=recent_rmse,
        )

    def compute_confidence_adjustment(
        self,
        base_confidence: float,
        value_range: float,
    ) -> float:
        """Compute confidence adjustment based on historical error.

        Args:
            base_confidence: Base confidence from stability
            value_range: Range of values (for normalization)

        Returns:
            Adjusted confidence (0.0 - 1.0)
        """
        metrics = self.get_metrics()

        if metrics is None or value_range < 1e-9:
            return base_confidence

        # Normalize recent MAE by value range
        normalized_error = metrics.recent_mae / value_range

        # Error penalty: high error → low confidence
        # normalized_error = 0.0 → penalty = 0.0 (no reduction)
        # normalized_error = 0.1 → penalty = 0.1
        # normalized_error = 0.5 → penalty = 0.5
        error_penalty = min(normalized_error, 0.5)

        adjusted = base_confidence * (1.0 - error_penalty)

        logger.debug(
            "taylor_confidence_adjusted",
            extra={
                "base_confidence": round(base_confidence, 4),
                "recent_mae": round(metrics.recent_mae, 4),
                "normalized_error": round(normalized_error, 4),
                "error_penalty": round(error_penalty, 4),
                "adjusted_confidence": round(adjusted, 4),
            },
        )

        return max(0.0, min(1.0, adjusted))

    def reset(self) -> None:
        """Reset all metrics."""
        self._errors.clear()
        self._total_squared_error = 0.0
        self._total_absolute_error = 0.0
        self._n_samples = 0
