"""Persistence Adapter — Phase 5 Scalability + Phase 6 Circuit Breaker.
Async persistence of regime state to external storage.
Ensures learning history survives pod restarts in Kubernetes.
Phase 6: Added Circuit Breaker for fault tolerance - falls back to 
"Modo Amnésico" (RAM-only) when persistence fails.
"""
from __future__ import annotations
import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from ..context_state_manager import SeriesState
logger = logging.getLogger(__name__)
@dataclass
class CircuitBreakerState:
    """Circuit breaker state for fault tolerance."""
    failures: int = 0
    last_failure: float = 0.0
    is_open: bool = False
    threshold: int = 5
    timeout_seconds: float = 30.0
    def record_failure(self) -> None:
        self.failures += 1
        self.last_failure = time.time()
        if self.failures >= self.threshold:
            self.is_open = True
            logger.warning("circuit_breaker_opened: persistence failing, switching to Modo Amnesico")
    def record_success(self) -> None:
        if self.is_open and time.time() - self.last_failure > self.timeout_seconds:
            self.is_open = False
            self.failures = 0
            logger.info("circuit_breaker_closed: persistence recovered")
    def can_attempt(self) -> bool:
        if not self.is_open:
            return True
        if time.time() - self.last_failure > self.timeout_seconds:
            self.is_open = False
            self.failures = 0
            return True
        return False

class PersistenceAdapter(ABC):
    """Abstract adapter for async state persistence."""
    @abstractmethod
    async def save_state(self, series_id: str, state: "SeriesState") -> bool:
        """Persist series state asynchronously."""
        ...
    @abstractmethod
    async def load_state(self, series_id: str) -> Optional["SeriesState"]:
        """Load series state from storage."""
        ...
    @abstractmethod
    async def delete_state(self, series_id: str) -> bool:
        """Delete series state (for cleanup)."""
        ...

class RedisPersistenceAdapter(PersistenceAdapter):
    """Redis-backed async persistence with Circuit Breaker. Fast recovery (<10ms), TTL cleanup."""
    def __init__(self, redis_client, ttl_seconds: int = 3600 * 24 * 7, key_prefix: str = "zenin:regime:") -> None:
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._prefix = key_prefix
        self._circuit = CircuitBreakerState()
    def _key(self, series_id: str) -> str:
        return f"{self._prefix}{series_id}"
    async def save_state(self, series_id: str, state: "SeriesState") -> bool:
        if not self._circuit.can_attempt():
            return False
        try:
            data = json.dumps({"regime": state.regime, "last_updated": state.last_updated, "prediction_count": state.prediction_count})
            await self._redis.setex(self._key(series_id), self._ttl, data)
            self._circuit.record_success()
            return True
        except Exception as e:
            self._circuit.record_failure()
            logger.warning(f"redis_save_failed: {e}")
            return False
    async def load_state(self, series_id: str) -> Optional["SeriesState"]:
        if not self._circuit.can_attempt():
            return None
        from ..context_state_manager import SeriesState
        try:
            data = await self._redis.get(self._key(series_id))
            if not data:
                return None
            parsed = json.loads(data)
            self._circuit.record_success()
            return SeriesState(regime=parsed.get("regime"), last_updated=parsed.get("last_updated", 0.0), prediction_count=parsed.get("prediction_count", 0))
        except Exception as e:
            self._circuit.record_failure()
            logger.warning(f"redis_load_failed: {e}")
            return None
    async def delete_state(self, series_id: str) -> bool:
        if not self._circuit.can_attempt():
            return False
        try:
            await self._redis.delete(self._key(series_id))
            self._circuit.record_success()
            return True
        except Exception as e:
            self._circuit.record_failure()
            logger.warning(f"redis_delete_failed: {e}")
            return False

class PostgresPersistenceAdapter(PersistenceAdapter):
    """PostgreSQL-backed async persistence with Circuit Breaker. Long-term history, analytics."""
    def __init__(self, async_pool, table_name: str = "zenin_regime_state") -> None:
        self._pool = async_pool
        self._table = table_name
        self._circuit = CircuitBreakerState()
    async def save_state(self, series_id: str, state: "SeriesState") -> bool:
        if not self._circuit.can_attempt():
            return False
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(f"""
                    INSERT INTO {self._table} (series_id, regime, last_updated, prediction_count, updated_at)
                    VALUES ($1, $2, $3, $4, NOW())
                    ON CONFLICT (series_id) DO UPDATE SET regime=$2, last_updated=$3, prediction_count=$4, updated_at=NOW()
                """, series_id, state.regime, state.last_updated, state.prediction_count)
            self._circuit.record_success()
            return True
        except Exception as e:
            self._circuit.record_failure()
            logger.warning(f"postgres_save_failed: {e}")
            return False
    async def load_state(self, series_id: str) -> Optional["SeriesState"]:
        if not self._circuit.can_attempt():
            return None
        from ..context_state_manager import SeriesState
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(f"SELECT regime, last_updated, prediction_count FROM {self._table} WHERE series_id=$1", series_id)
                if not row:
                    return None
                self._circuit.record_success()
                return SeriesState(regime=row["regime"], last_updated=row["last_updated"], prediction_count=row["prediction_count"])
        except Exception as e:
            self._circuit.record_failure()
            logger.warning(f"postgres_load_failed: {e}")
            return None
    async def delete_state(self, series_id: str) -> bool:
        if not self._circuit.can_attempt():
            return False
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(f"DELETE FROM {self._table} WHERE series_id=$1", series_id)
            self._circuit.record_success()
            return True
        except Exception as e:
            self._circuit.record_failure()
            logger.warning(f"postgres_delete_failed: {e}")
            return False

class HybridPersistenceAdapter(PersistenceAdapter):
    """Hybrid: Redis (fast) + Postgres (durable). Write-through to both, read Redis first."""
    def __init__(self, redis_adapter: RedisPersistenceAdapter, postgres_adapter: PostgresPersistenceAdapter) -> None:
        self._redis = redis_adapter
        self._postgres = postgres_adapter
    async def save_state(self, series_id: str, state: "SeriesState") -> bool:
        results = await asyncio.gather(self._redis.save_state(series_id, state), self._postgres.save_state(series_id, state), return_exceptions=True)
        return any(r is True for r in results if not isinstance(r, Exception))
    async def load_state(self, series_id: str) -> Optional["SeriesState"]:
        state = await self._redis.load_state(series_id)
        if state:
            return state
        state = await self._postgres.load_state(series_id)
        if state:
            await self._redis.save_state(series_id, state)
        return state
    async def delete_state(self, series_id: str) -> bool:
        results = await asyncio.gather(self._redis.delete_state(series_id), self._postgres.delete_state(series_id), return_exceptions=True)
        return any(r is True for r in results if not isinstance(r, Exception))
