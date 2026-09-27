"""Streaming adaptive normalizer for continuous states and multi-sensor signals.

Maintains running mean and variance online via Welford's algorithm to convert
arbitrary physical sensor dimensions into dimensionless standard innovations N(0, 1).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence, cast
import numpy as np

from core.parameters.numerical_constants import EPSILON


class StreamingNormalizer:
    """Online streaming z-score normalizer with scale protection and bounded state."""

    def __init__(
        self,
        dimension: int = 1,
        clip_range: float = 6.0,
        min_variance: float = 1e-6,
        warmup_samples: int = 5,
    ) -> None:
        self.dimension = max(1, dimension)
        self.clip_range = float(clip_range)
        self.min_variance = float(min_variance)
        self.warmup_samples = max(2, warmup_samples)

        self._count: int = 0
        self._mean: np.ndarray = np.zeros(self.dimension, dtype=np.float64)
        self._m2: np.ndarray = np.zeros(self.dimension, dtype=np.float64)

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> np.ndarray:
        return cast(np.ndarray, self._mean.copy())

    @property
    def variance(self) -> np.ndarray:
        if self._count < 2:
            return np.ones(self.dimension, dtype=np.float64)
        return np.maximum(self.min_variance, self._m2 / (self._count - 1))

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.variance)

    def _ensure_dimension(self, arr: np.ndarray) -> np.ndarray:
        flat = arr.astype(np.float64).flatten()
        if flat.size != self.dimension:
            if self._count == 0:
                self.dimension = flat.size
                self._mean = np.zeros(self.dimension, dtype=np.float64)
                self._m2 = np.zeros(self.dimension, dtype=np.float64)
            else:
                raise ValueError(f"Dim mismatch: expected {self.dimension}, got {flat.size}")
        return flat

    def update(self, x: float | Sequence[float] | np.ndarray) -> np.ndarray:
        """Update online statistics with observation x and return normalized z-score."""
        arr = self._ensure_dimension(np.asarray(x, dtype=np.float64))
        self._count += 1
        delta = arr - self._mean
        self._mean += delta / self._count
        delta2 = arr - self._mean
        self._m2 += delta * delta2

        if self._count < self.warmup_samples:
            return np.zeros_like(arr)

        return self.transform(arr)

    def transform(self, x: float | Sequence[float] | np.ndarray) -> np.ndarray:
        """Project observation into dimensionless z-score space without updating statistics."""
        arr = self._ensure_dimension(np.asarray(x, dtype=np.float64))
        if self._count < self.warmup_samples:
            return np.zeros_like(arr)
        effective_std = np.maximum(self.std, math.sqrt(self.min_variance) + EPSILON.DIVISION)
        z = (arr - self._mean) / effective_std
        if self.clip_range > 0:
            z = np.clip(z, -self.clip_range, self.clip_range)
        return cast(np.ndarray, z)

    def inverse_transform(self, z: float | Sequence[float] | np.ndarray) -> np.ndarray:
        """Reconstruct original physical scale from normalized z-score."""
        arr = self._ensure_dimension(np.asarray(z, dtype=np.float64))
        return cast(np.ndarray, arr * self.std + self._mean)

    def reset(self) -> None:
        """Reset running statistics to cold start."""
        self._count = 0
        self._mean.fill(0.0)
        self._m2.fill(0.0)

    def export_state(self) -> Dict[str, Any]:
        """Serialize state for persistent snapshotting."""
        return {
            "schema_version": 1,
            "dimension": self.dimension,
            "count": self._count,
            "mean": self._mean.tolist(),
            "m2": self._m2.tolist(),
            "clip_range": self.clip_range,
            "min_variance": self.min_variance,
            "warmup_samples": self.warmup_samples,
        }

    def import_state(self, payload: Dict[str, Any]) -> None:
        """Restore running state from snapshot."""
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("Invalid normalizer payload or schema_version mismatch")
        self.dimension = int(payload["dimension"])
        self._count = int(payload["count"])
        self._mean = np.array(payload["mean"], dtype=np.float64)
        self._m2 = np.array(payload["m2"], dtype=np.float64)
        self.clip_range = float(payload.get("clip_range", self.clip_range))
        self.min_variance = float(payload.get("min_variance", self.min_variance))
        self.warmup_samples = int(payload.get("warmup_samples", self.warmup_samples))
