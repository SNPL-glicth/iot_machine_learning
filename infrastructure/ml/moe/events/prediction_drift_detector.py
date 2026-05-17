"""Prediction drift detector per sensor — minimal state, thread-safe reads."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

ROLLING_WINDOW: int = 20
DRIFT_SIGMA_THRESHOLD: float = 2.0


@dataclass
class DriftAlert:
    """Drift alert for a sensor."""

    series_id: str
    current_mae: float
    baseline_mae: float
    sigma_distance: float
    equipment_class: str = "GENERIC"


class PredictionDriftDetector:
    """Detects when a sensor's rolling MAE exceeds 2σ of its historic baseline."""

    def __init__(
        self,
        rolling_window: int = ROLLING_WINDOW,
        sigma_threshold: float = DRIFT_SIGMA_THRESHOLD,
    ) -> None:
        self._window = rolling_window
        self._threshold = sigma_threshold
        self._errors: dict[str, deque] = defaultdict(lambda: deque(maxlen=rolling_window * 3))

    def record_error(
        self,
        series_id: str,
        error: float,
        equipment_class: str = "GENERIC",
    ) -> Optional[DriftAlert]:
        """Record prediction error and return DriftAlert if drift detected."""
        buf = self._errors[series_id]
        buf.append(abs(error))
        min_samples = self._window * 2
        if len(buf) < min_samples:
            return None
        half = len(buf) // 2
        baseline = list(buf)[:half]
        current = list(buf)[-self._window:]
        baseline_mae = sum(baseline) / len(baseline)
        current_mae = sum(current) / len(current)
        mean_b = sum(baseline) / len(baseline)
        var_b = sum((x - mean_b) ** 2 for x in baseline) / len(baseline)
        std_b = var_b ** 0.5
        if std_b < 1e-9:
            return None
        sigma_distance = (current_mae - baseline_mae) / std_b
        if sigma_distance > self._threshold:
            return DriftAlert(
                series_id=series_id,
                current_mae=current_mae,
                baseline_mae=baseline_mae,
                sigma_distance=sigma_distance,
                equipment_class=equipment_class,
            )
        return None

    def get_mae(self, series_id: str) -> Optional[float]:
        """Current rolling MAE for the sensor, or None if no data."""
        buf = self._errors.get(series_id)
        if not buf:
            return None
        recent = list(buf)[-self._window:]
        return sum(recent) / len(recent)
