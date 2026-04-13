"""Taylor coefficient cache — FASE 2.

Caches Taylor coefficients to avoid recalculating derivatives every prediction.

Fixes CRIT-1: Taylor NO aprende — now caches coefficients with TTL.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

from .types import TaylorCoefficients

logger = logging.getLogger(__name__)


@dataclass
class CachedCoefficients:
    """Cached Taylor coefficients with metadata."""

    coefficients: TaylorCoefficients
    cached_at: float
    window_hash: str
    n_points: int
    dt: float


class TaylorCoefficientCache:
    """In-memory cache for Taylor coefficients with TTL.

    Attributes:
        _cache: Dict mapping series_id -> CachedCoefficients
        _ttl_seconds: Time-to-live for cached coefficients
        _hits: Cache hit counter
        _misses: Cache miss counter
    """

    def __init__(self, ttl_seconds: int = 300):
        """Initialize cache.

        Args:
            ttl_seconds: TTL for cached coefficients (default 5 minutes)
        """
        self._cache: Dict[str, CachedCoefficients] = {}
        self._ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0

    def get(
        self,
        series_id: str,
        window_hash: str,
    ) -> Optional[TaylorCoefficients]:
        """Get cached coefficients if valid.

        Args:
            series_id: Series identifier
            window_hash: Hash of input window

        Returns:
            Cached coefficients or None if expired/not found
        """
        cached = self._cache.get(series_id)

        if cached is None:
            self._misses += 1
            return None

        # Check if expired
        age = time.time() - cached.cached_at
        if age > self._ttl_seconds:
            del self._cache[series_id]
            self._misses += 1
            logger.debug(
                "taylor_cache_expired",
                extra={
                    "series_id": series_id,
                    "age_seconds": int(age),
                },
            )
            return None

        # Check if window changed
        if cached.window_hash != window_hash:
            self._misses += 1
            return None

        self._hits += 1
        logger.debug(
            "taylor_cache_hit",
            extra={
                "series_id": series_id,
                "age_seconds": int(age),
            },
        )

        return cached.coefficients

    def put(
        self,
        series_id: str,
        coefficients: TaylorCoefficients,
        window_hash: str,
        n_points: int,
        dt: float,
    ) -> None:
        """Cache coefficients.

        Args:
            series_id: Series identifier
            coefficients: Taylor coefficients
            window_hash: Hash of input window
            n_points: Number of points in window
            dt: Time step
        """
        self._cache[series_id] = CachedCoefficients(
            coefficients=coefficients,
            cached_at=time.time(),
            window_hash=window_hash,
            n_points=n_points,
            dt=dt,
        )

        logger.debug(
            "taylor_cache_put",
            extra={
                "series_id": series_id,
                "n_points": n_points,
            },
        )

    def invalidate(self, series_id: str) -> None:
        """Invalidate cache for series.

        Args:
            series_id: Series identifier
        """
        if series_id in self._cache:
            del self._cache[series_id]
            logger.debug(
                "taylor_cache_invalidated",
                extra={"series_id": series_id},
            )

    def clear(self) -> None:
        """Clear entire cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_metrics(self) -> dict:
        """Get cache metrics.

        Returns:
            Dict with metrics
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "cached_series": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
            "ttl_seconds": self._ttl_seconds,
        }

    @staticmethod
    def compute_window_hash(values: list, timestamps: Optional[list] = None) -> str:
        """Compute hash of input window.

        Args:
            values: Input values
            timestamps: Optional timestamps

        Returns:
            Hash string
        """
        import hashlib
        import json

        data = json.dumps({
            "values": values,
            "timestamps": timestamps,
        }, sort_keys=True)

        return hashlib.sha256(data.encode()).hexdigest()[:16]
