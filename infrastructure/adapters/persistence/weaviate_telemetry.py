"""Asynchronous batching Weaviate persistence layer for Zephyr telemetry and cognitive memory."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
import json
import logging
from typing import Any, Mapping, Optional
import uuid
import aiohttp

logger = logging.getLogger(__name__)


class WeaviateTelemetryStore:
    """Asynchronous, batching Weaviate store for telemetry streaming and cognitive memory querying."""

    def __init__(
        self,
        url: str = "http://localhost:8080",
        batch_size: int = 50,
        flush_interval_sec: float = 2.0,
        max_queue_size: int = 5000,
    ) -> None:
        self._base_url = url.rstrip("/")
        self._batch_size = batch_size
        self._flush_interval = flush_interval_sec
        self._queue: deque[dict[str, Any]] = deque(maxlen=max_queue_size)
        self._session: Optional[aiohttp.ClientSession] = None
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=5.0)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def start(self) -> None:
        """Starts the background batch flush loop."""
        if not self._running:
            self._running = True
            self._flush_task = asyncio.create_task(self._batch_flusher_loop())

    async def _batch_flusher_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self._flush_interval)
                await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Background Weaviate flush exception: %s", e)

    async def flush(self) -> int:
        """Flushes buffered objects to Weaviate in a single batch POST."""
        if not self._queue:
            return 0
        batch: list[dict[str, Any]] = []
        async with self._lock:
            while self._queue and len(batch) < self._batch_size:
                batch.append(self._queue.popleft())
        if not batch:
            return 0
        try:
            session = await self._ensure_session()
            async with session.post(
                f"{self._base_url}/v1/batch/objects",
                json={"objects": batch},
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status not in (200, 201):
                    txt = await resp.text()
                    logger.debug("Weaviate batch write returned status %d: %s", resp.status, txt)
                return len(batch)
        except Exception as e:
            logger.debug("Weaviate batch flush network error: %s", e)
            return 0

    def log_telemetry(self, telemetry_state: Mapping[str, Any]) -> None:
        """Enqueues telemetry snapshot non-blockingly."""
        props = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": str(telemetry_state.get("symbol", "UNKNOWN")),
            "midPrice": float(telemetry_state.get("mid_price", 0.0)),
            "phiMoe": float(telemetry_state.get("phi_moe", 0.0)),
            "cvar": float(telemetry_state.get("cvar", 0.0)),
            "lambdaT": float(telemetry_state.get("lambda_t", 0.0)),
            "action": str(telemetry_state.get("action", "HOLD")),
            "reason": str(telemetry_state.get("reason", "")),
            "engineRationale": json.dumps(telemetry_state.get("decision_rationale", {})),
        }
        self._queue.append({"class": "ZeninTelemetry", "id": str(uuid.uuid4()), "properties": props})
        self._maybe_trigger_immediate_flush()

    def log_execution(self, execution_data: Mapping[str, Any]) -> None:
        """Enqueues execution event non-blockingly."""
        props = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": str(execution_data.get("symbol", "UNKNOWN")),
            "side": str(execution_data.get("side", "")),
            "qty": float(execution_data.get("qty", 0.0)),
            "entryPrice": float(execution_data.get("entry_price", 0.0)),
            "pnlUsd": float(execution_data.get("pnl_usd", 0.0)),
            "status": str(execution_data.get("status", "SUBMITTED")),
        }
        self._queue.append({"class": "ZeninExecution", "id": str(uuid.uuid4()), "properties": props})
        self._maybe_trigger_immediate_flush()

    def log_config_change(self, config_snapshot: Mapping[str, Any], source: str = "dashboard") -> None:
        """Enqueues sanitized config audit event."""
        sanitized = {k: ("***REDACTED***" if "key" in k or "secret" in k else v) for k, v in config_snapshot.items()}
        props = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "configSnapshot": json.dumps(sanitized),
        }
        self._queue.append({"class": "ZeninConfigAudit", "id": str(uuid.uuid4()), "properties": props})
        self._maybe_trigger_immediate_flush()

    async def query_recent_telemetry(self, symbol: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Queries recent ZeninTelemetry records from Weaviate for dashboard visualization."""
        try:
            session = await self._ensure_session()
            url = f"{self._base_url}/v1/objects?class=ZeninTelemetry&limit={limit}"
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    objs = data.get("objects", [])
                    if symbol:
                        objs = [o for o in objs if o.get("properties", {}).get("symbol") == symbol]
                    return [o.get("properties", {}) for o in objs]
        except Exception as e:
            logger.debug("Weaviate query_recent_telemetry error: %s", e)
        return []

    async def query_recent_executions(self, symbol: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Queries recent ZeninExecution records from Weaviate."""
        try:
            session = await self._ensure_session()
            url = f"{self._base_url}/v1/objects?class=ZeninExecution&limit={limit}"
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    objs = data.get("objects", [])
                    if symbol:
                        objs = [o for o in objs if o.get("properties", {}).get("symbol") == symbol]
                    return [o.get("properties", {}) for o in objs]
        except Exception as e:
            logger.debug("Weaviate query_recent_executions error: %s", e)
        return []

    def _maybe_trigger_immediate_flush(self) -> None:
        if len(self._queue) >= self._batch_size:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.flush())
            except RuntimeError:
                pass

    async def flush_and_close(self) -> None:
        """Drains pending records and closes the session gracefully."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
        while self._queue:
            await self.flush()
        if self._session and not self._session.closed:
            await self._session.close()
