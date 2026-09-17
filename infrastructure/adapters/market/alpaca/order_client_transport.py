"""HTTP Transport layer with rate limiting and backoff for Alpaca."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any, Dict, Optional, TypeVar, cast

import random
import aiohttp
import numpy as np

from iot_machine_learning.infrastructure.adapters.market.alpaca.order_client_constants import OrderClientConstants
from iot_machine_learning.infrastructure.adapters.market.alpaca.rate_limiter import RateLimiter
from iot_machine_learning.infrastructure.adapters.market.zephyr.resilience.circuit_breaker import (
    CircuitBreaker, CircuitBreakerOpenError,
)

logger = logging.getLogger(__name__)


_T = TypeVar("_T", bound="AlpacaOrderTransport")


class AlpacaOrderTransport:
    """Manejo de conexión HTTP, rate limiting, retry y métricas para Alpaca."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: Optional[str] = None,
        *,
        data_feed: str = "iex",
        recv_window: int = 5000,
        max_retries: int = 3,
        base_retry_delay: float = 0.5,
        max_retry_delay: float = 10.0,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url or OrderClientConstants.BASE_URL
        self.data_feed_base_url = "https://data.alpaca.markets/v2"
        self.data_feed = data_feed
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        self.max_retry_delay = max_retry_delay

        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter = RateLimiter()
        self._circuit_breaker = CircuitBreaker(name="alpaca_rest_api", failure_threshold=5, recovery_timeout=20.0)
        self._closed = False
        self._request_count = 0
        self._error_count = 0
        self._latencies: deque = deque(maxlen=1000)

    async def __aenter__(self: _T) -> _T:
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=20)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    "APCA-API-KEY-ID": self.api_key,
                    "APCA-API-SECRET-KEY": self.api_secret,
                },
            )

    async def close(self) -> None:
        """Cierra la sesión HTTP."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._closed = True

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        weight: int = 1,
        use_data_api: bool = False,
    ) -> Dict[str, Any]:
        """Ejecuta request HTTP con Circuit Breaker, rate limiting, exponential backoff con jitter."""
        if not self._circuit_breaker.can_execute():
            raise CircuitBreakerOpenError(f"Broker REST API circuit breaker is OPEN for endpoint {endpoint}")

        await self._rate_limiter.acquire(weight)
        await self._ensure_session()
        assert self._session is not None
        base = self.data_feed_base_url if use_data_api else self.base_url
        url = f"{base}{endpoint}"
        headers = {"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.api_secret}
        if method in ("POST", "PUT", "PATCH"): headers["Content-Type"] = "application/json"

        last_exception = None
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                async with self._session.request(method, url, params=params, json=json_data, headers=headers) as resp:
                    self._latencies.append(time.perf_counter() - start)
                    if resp.status in (200, 201, 207):
                        self._request_count += 1
                        self._circuit_breaker.record_success()
                        return cast(Dict[str, Any], await resp.json())
                    if resp.status == 204:
                        self._request_count += 1
                        self._circuit_breaker.record_success()
                        return {"status": "success", "code": 204}
                    if resp.status == 429:
                        retry_after = float(resp.headers.get("Retry-After", 1.0)) + random.uniform(0.1, 0.4)
                        logger.warning("Rate limited (429) on %s %s, backoff %.2fs", method, endpoint, retry_after)
                        await asyncio.sleep(retry_after)
                        continue
                    err_txt = await resp.text()
                    if resp.status in (500, 502, 503, 504):
                        last_exception = RuntimeError(f"Server error HTTP {resp.status}: {err_txt}")
                        logger.warning("Transient broker error HTTP %d (attempt %d/%d)", resp.status, attempt + 1, self.max_retries + 1)
                    elif resp.status in (401, 403):
                        self._circuit_breaker.record_failure()
                        raise RuntimeError(f"Auth/Permission error {resp.status}: {err_txt}")
                    else:
                        raise RuntimeError(f"HTTP {resp.status}: {err_txt}")
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                last_exception = e
                logger.warning("Network I/O error on %s %s: %s (attempt %d)", method, endpoint, e, attempt + 1)

            if attempt < self.max_retries:
                jitter = random.uniform(0.05, 0.25)
                delay = min(self.base_retry_delay * (2 ** attempt), self.max_retry_delay) + jitter
                await asyncio.sleep(delay)

        self._error_count += 1
        self._circuit_breaker.record_failure(last_exception)
        raise RuntimeError(f"Request {method} {endpoint} failed after {self.max_retries + 1} attempts: {last_exception}")

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(1, self._request_count),
            "circuit_breaker": self._circuit_breaker.get_metrics(),
            "avg_latency_ms": np.mean(self._latencies) * 1000 if self._latencies else 0,
            "p50_latency_ms": np.percentile(self._latencies, 50) * 1000 if self._latencies else 0,
            "p99_latency_ms": np.percentile(self._latencies, 99) * 1000 if self._latencies else 0,
            "session_closed": self._closed,
        }

