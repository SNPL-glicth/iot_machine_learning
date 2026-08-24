"""MLStateStore adapters: in-memory fallback and Redis-backed snapshot store.

Snapshots are single JSON documents keyed by engine id, so a load is atomic:
either the full engine snapshot or nothing. Corrupt payloads surface as None
(cold start) instead of exceptions at the transport layer; schema validation
remains the engine's responsibility.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 48 * 3600


class InMemoryMLStateStore:
    """Process-local store. Useful for tests and as graceful fallback."""

    def __init__(self) -> None:
        self._snapshots: Dict[str, Dict[str, Any]] = {}

    def save(self, engine_id: str, payload: Dict[str, Any]) -> bool:
        self._snapshots[engine_id] = payload
        return True

    def load(self, engine_id: str) -> Optional[Dict[str, Any]]:
        payload = self._snapshots.get(engine_id)
        return dict(payload) if payload is not None else None

    def delete(self, engine_id: str) -> bool:
        return self._snapshots.pop(engine_id, None) is not None


class RedisMLStateStore:
    """Redis-backed store with TTL. Single key per engine snapshot."""

    def __init__(
        self,
        redis_url: str,
        key_prefix: str = "mlstate",
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ):
        import redis  # Lazy: keeps this module importable without redis.

        self._key_prefix = key_prefix
        self._ttl_seconds = ttl_seconds
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    def _key(self, engine_id: str) -> str:
        return f"{self._key_prefix}:{engine_id}"

    def save(self, engine_id: str, payload: Dict[str, Any]) -> bool:
        try:
            encoded = json.dumps(payload, default=str)
            self._client.set(self._key(engine_id), encoded, ex=self._ttl_seconds)
            return True
        except Exception as exc:  # Never block the hot path on storage errors.
            logger.warning("MLStateStore save failed for %s: %s", engine_id, exc)
            return False

    def load(self, engine_id: str) -> Optional[Dict[str, Any]]:
        try:
            raw = self._client.get(self._key(engine_id))
            if raw is None:
                return None
            decoded = json.loads(raw)
            return decoded if isinstance(decoded, dict) else None
        except Exception as exc:
            logger.warning("MLStateStore load failed for %s: %s", engine_id, exc)
            return None

    def delete(self, engine_id: str) -> bool:
        try:
            return bool(self._client.delete(self._key(engine_id)))
        except Exception as exc:
            logger.warning("MLStateStore delete failed for %s: %s", engine_id, exc)
            return False


def create_state_store(config: Dict[str, Any]):
    """Build a state store from service config.

    config keys:
        backend: None | "memory" | "redis"
        redis_url: str (required for redis)
        redis_prefix / ttl_seconds: optional tuning
    """
    backend = (config or {}).get("backend")
    if backend is None:
        return None
    if backend == "memory":
        return InMemoryMLStateStore()
    if backend == "redis":
        redis_url = (config or {}).get("redis_url")
        if not redis_url:
            raise ValueError("redis backend requires 'redis_url'")
        return RedisMLStateStore(
            redis_url=redis_url,
            key_prefix=(config or {}).get("redis_prefix", "mlstate"),
            ttl_seconds=int((config or {}).get("ttl_seconds", DEFAULT_TTL_SECONDS)),
        )
    raise ValueError(f"Unknown ML state store backend: {backend!r}")
