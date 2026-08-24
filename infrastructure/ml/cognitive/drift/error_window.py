"""Rolling error window with statistics."""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, List

import numpy as np

from core.parameters.numerical_constants import EPSILON


def safe_mean(values: Deque[float]) -> float:
    """Mean with NaN/Inf guards."""
    if not values:
        return 0.0
    valid = [v for v in values if math.isfinite(v)]
    if not valid:
        return 0.0
    return float(np.mean(valid))


def safe_std(values: Deque[float]) -> float:
    """Std with NaN/Inf guards and small-sample protection."""
    if len(values) < 2:
        return 0.0
    valid = [v for v in values if math.isfinite(v)]
    if len(valid) < 2:
        return 0.0
    return float(np.std(valid, ddof=1))


class ErrorWindow:
    """Rolling window of absolute errors with running statistics."""
    
    def __init__(self, window_size: int = 100):
        if window_size < 10:
            raise ValueError(f"window_size must be ≥ 10, got {window_size}")
        self._window_size = window_size
        self._errors: Deque[float] = deque(maxlen=window_size)
        self._rolling_mae = 0.0
        self._rolling_std = 0.0
        self._n_updates = 0
    
    def add(self, abs_error: float) -> None:
        self._errors.append(abs_error)
        self._n_updates += 1
        self._rolling_mae = safe_mean(self._errors)
        self._rolling_std = safe_std(self._errors)
    
    def get_last_error(self) -> float:
        return self._errors[-1] if self._errors else 0.0
    
    @property
    def rolling_mae(self) -> float:
        return self._rolling_mae
    
    @property
    def rolling_std(self) -> float:
        return self._rolling_std
    
    @property
    def n_updates(self) -> int:
        return self._n_updates
    
    @property
    def errors(self) -> Deque[float]:
        return self._errors
    
    def clear(self) -> None:
        self._errors.clear()
        self._rolling_mae = 0.0
        self._rolling_std = 0.0
        self._n_updates = 0

    # ------------------------------------------------------------------
    # Persistence (StatePersistable contract)
    # ------------------------------------------------------------------

    STATE_SCHEMA_VERSION = 1

    def export_state(self) -> dict:
        """Serialize window contents and running statistics."""
        return {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "window_size": self._window_size,
            "errors": [float(e) for e in self._errors],
            "rolling_mae": self._rolling_mae,
            "rolling_std": self._rolling_std,
            "n_updates": self._n_updates,
        }

    def import_state(self, payload: dict) -> None:
        """Restore window contents and running statistics."""
        if not isinstance(payload, dict) or "schema_version" not in payload:
            raise ValueError("ErrorWindow payload missing schema_version")
        if payload["schema_version"] != self.STATE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported ErrorWindow schema: {payload['schema_version']}")
        errors_raw = payload.get("errors", [])
        if not isinstance(errors_raw, list):
            raise ValueError("ErrorWindow payload 'errors' must be a list")
        window_size = int(payload.get("window_size", self._window_size))
        if window_size != self._window_size:
            raise ValueError(
                f"ErrorWindow size mismatch: snapshot={window_size}, live={self._window_size}"
            )
        if len(errors_raw) > self._window_size:
            raise ValueError("ErrorWindow snapshot exceeds capacity")
        for value in errors_raw:
            float(value)
        self.clear()
        for value in errors_raw:
            self._errors.append(float(value))
        self._rolling_mae = float(payload.get("rolling_mae", 0.0))
        self._rolling_std = float(payload.get("rolling_std", 0.0))
        self._n_updates = int(payload.get("n_updates", 0))