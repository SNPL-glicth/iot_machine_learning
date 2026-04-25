"""SeriesValuesStore — Redis-backed rolling buffer of raw sensor values.

Per ``series_id`` persists the last ``max_values`` finite floats into a
Redis list at ``series_values:{series_id}`` with TTL refreshed on every
write. Supplies ``(lower, upper)`` mean±kσ bounds for the sanitize
phase.

Design (IMP-1, user-confirmed):
    * **Redis-only**. When ``redis_client`` is ``None`` or Redis raises,
      all public methods become inert (``append`` no-op, ``get_recent``
      returns ``[]``, ``get_bounds`` returns ``None``).
    * **Never raises.** Every Redis call is wrapped; exceptions are
      logged at WARNING and swallowed.
    * **SRP**. Owns value history I/O only. Statistics are derived on
      read; no caching.
    * **NaN/Inf rejected on write** — the buffer only ever contains
      finite floats.
"""

from __future__ import annotations

import logging
import math
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


_DEFAULT_PREFIX: str = "series_values"
_DEFAULT_TTL_SECONDS: int = 7 * 24 * 3600  # 7 days
_DEFAULT_MAX_VALUES: int = 500
_DEFAULT_MIN_SAMPLES: int = 20


class SeriesValuesStore:
    """Redis-backed rolling buffer of raw sensor values."""

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        *,
        key_prefix: str = _DEFAULT_PREFIX,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        max_values: int = _DEFAULT_MAX_VALUES,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be > 0, got {ttl_seconds}")
        if max_values <= 0:
            raise ValueError(f"max_values must be > 0, got {max_values}")
        if not key_prefix:
            raise ValueError("key_prefix must be non-empty")

        self._redis = redis_client
        self._prefix = key_prefix
        self._ttl = int(ttl_seconds)
        self._max = int(max_values)

    # -- public API ---------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True iff a Redis client was supplied."""
        return self._redis is not None

    def append(self, series_id: str, value: float) -> None:
        """Push a finite float to the series buffer. Never raises.

        Non-finite values and non-numeric inputs are silently dropped.
        """
        if self._redis is None or not series_id:
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(v):
            return
        try:
            key = self._key(series_id)
            pipe = self._redis.pipeline()
            pipe.rpush(key, repr(v))
            pipe.ltrim(key, -self._max, -1)
            pipe.expire(key, self._ttl)
            pipe.execute()
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "series_values_redis_write_failed",
                extra={"series_id": series_id, "error": str(exc)},
            )

    def append_many(self, series_id: str, values: List[float]) -> None:
        """Push several finite floats in a single Redis round-trip."""
        if self._redis is None or not series_id or not values:
            return
        coerced: List[str] = []
        for v in values:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv):
                coerced.append(repr(fv))
        if not coerced:
            return
        try:
            key = self._key(series_id)
            pipe = self._redis.pipeline()
            pipe.rpush(key, *coerced)
            pipe.ltrim(key, -self._max, -1)
            pipe.expire(key, self._ttl)
            pipe.execute()
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "series_values_redis_write_failed",
                extra={"series_id": series_id, "error": str(exc)},
            )

    def get_recent(self, series_id: str, n: Optional[int] = None) -> List[float]:
        """Return up to ``n`` most recent finite values (chronological)."""
        if self._redis is None or not series_id:
            return []
        try:
            key = self._key(series_id)
            raw = self._redis.lrange(key, 0, -1)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "series_values_redis_read_failed",
                extra={"series_id": series_id, "error": str(exc)},
            )
            return []
        values = self._decode(raw)
        if n is not None and n > 0 and len(values) > n:
            return values[-n:]
        return values

    def get_bounds(
        self,
        series_id: str,
        sigma_multiplier: float = 6.0,
        min_samples: int = _DEFAULT_MIN_SAMPLES,
    ) -> Optional[Tuple[float, float]]:
        """Return ``(lower, upper)`` = ``mean ± sigma_multiplier*std``.

        Returns ``None`` when fewer than ``min_samples`` values are
        available or when the std is zero.
        """
        if sigma_multiplier <= 0 or min_samples <= 0:
            return None
        values = self.get_recent(series_id)
        if len(values) < min_samples:
            return None
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        if variance <= 0.0:
            return None
        std = math.sqrt(variance)
        return (mean - sigma_multiplier * std, mean + sigma_multiplier * std)

    def reset(self, series_id: str) -> None:
        """Delete the buffer for the series. Never raises."""
        if self._redis is None or not series_id:
            return
        try:
            self._redis.delete(self._key(series_id))
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "series_values_redis_delete_failed",
                extra={"series_id": series_id, "error": str(exc)},
            )

    # -- helpers ------------------------------------------------------

    def _key(self, series_id: str) -> str:
        return f"{self._prefix}:{series_id}"

    @staticmethod
    def _decode(raw: List[Any]) -> List[float]:
        out: List[float] = []
        for item in raw:
            val = item.decode() if isinstance(item, (bytes, bytearray)) else item
            try:
                fv = float(val)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv):
                out.append(fv)
        return out
