"""Redis window persistence models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class PersistedWindow:
    """Persisted window in Redis."""
    sensor_id: int
    values: List[float]
    timestamps: List[float]
    last_updated: float