"""HyperparameterAdaptor — Redis-backed per-engine hyperparameter store.

Per ``(series_id, engine_name)`` persists a small dictionary of numeric
hyperparameters (e.g. Taylor ``order``, Statistical ``alpha``/``beta``)
into a Redis Hash at ``engine_hyperparams:{series_id}:{engine_name}``
with TTL refreshed on every write. Values are stored as strings and
coerced back to ``float`` on load.

Design constraints (IMP-4c, user-confirmed):
    * **Redis-only**. No in-memory fallback. When ``redis_client`` is
      ``None`` or any Redis call raises, the adaptor is **inert**:
      ``load`` returns ``None``; ``save``/``reset`` are no-ops. Engines
      must fall back to their hardcoded defaults.
    * **Never raises**. All public methods swallow exceptions and log
      at WARNING level.
    * **Sole source of truth**. Replaces ``StatisticalParamsRepository``
      for statistical params and any engine-local persistence.
    * **SRP**. Owns hyperparameter I/O only.

This module is intentionally minimal — validation of individual
hyperparameter keys/ranges is the engine's responsibility.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


_DEFAULT_PREFIX: str = "engine_hyperparams"
_DEFAULT_TTL_SECONDS: int = 7 * 24 * 3600  # 7 days


class HyperparameterAdaptor:
    """Redis-only hyperparameter store (inert when Redis is absent)."""

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        *,
        key_prefix: str = _DEFAULT_PREFIX,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be > 0, got {ttl_seconds}")
        if not key_prefix:
            raise ValueError("key_prefix must be a non-empty string")

        self._redis = redis_client
        self._prefix = key_prefix
        self._ttl = int(ttl_seconds)

    # -- public API ---------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True iff a Redis client was supplied. Inert adaptors return False."""
        return self._redis is not None

    def load(self, series_id: str, engine_name: str) -> Optional[Dict[str, float]]:
        """Return the stored hyperparameters or ``None``.

        Returns ``None`` when the adaptor is inert, when the hash does
        not exist, or when Redis raises. Only finite float coercions
        are kept — malformed fields are silently dropped.
        """
        if self._redis is None:
            return None
        if not series_id or not engine_name:
            return None
        try:
            raw = self._redis.hgetall(self._key(series_id, engine_name))
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "hyperparams_redis_read_failed",
                extra={"series_id": series_id, "engine": engine_name, "error": str(exc)},
            )
            return None
        if not raw:
            return None
        return self._decode(raw)

    def save(
        self,
        series_id: str,
        engine_name: str,
        params: Dict[str, float],
    ) -> None:
        """Persist ``params`` to Redis. Never raises; no-op when inert.

        Non-numeric or non-finite values are silently skipped. When the
        resulting mapping is empty nothing is written.
        """
        if self._redis is None:
            return
        if not series_id or not engine_name or not params:
            return
        mapping = self._encode(params)
        if not mapping:
            return
        try:
            key = self._key(series_id, engine_name)
            pipe = self._redis.pipeline()
            pipe.hset(key, mapping=mapping)
            pipe.expire(key, self._ttl)
            pipe.execute()
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "hyperparams_redis_write_failed",
                extra={"series_id": series_id, "engine": engine_name, "error": str(exc)},
            )

    def reset(self, series_id: str, engine_name: str) -> None:
        """Delete the stored hyperparameters. Never raises; no-op when inert."""
        if self._redis is None:
            return
        if not series_id or not engine_name:
            return
        try:
            self._redis.delete(self._key(series_id, engine_name))
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "hyperparams_redis_delete_failed",
                extra={"series_id": series_id, "engine": engine_name, "error": str(exc)},
            )

    # -- helpers ------------------------------------------------------

    def _key(self, series_id: str, engine_name: str) -> str:
        return f"{self._prefix}:{series_id}:{engine_name}"

    @staticmethod
    def _encode(params: Dict[str, float]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for k, v in params.items():
            if not isinstance(k, str) or not k:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv != fv or fv in (float("inf"), float("-inf")):  # NaN / inf
                continue
            out[k] = repr(fv)
        return out

    @staticmethod
    def _decode(raw: Dict[Any, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in raw.items():
            key = k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
            val = v.decode() if isinstance(v, (bytes, bytearray)) else v
            try:
                fv = float(val)
            except (TypeError, ValueError):
                continue
            if fv != fv or fv in (float("inf"), float("-inf")):
                continue
            out[key] = fv
        return out
